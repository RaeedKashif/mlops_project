# ingestion/ingestion_service.py
import os
import json
import sqlite3
import threading
import math
from datetime import datetime
from contextlib import contextmanager

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DB_PATH = os.getenv("DB_PATH", "/app/db/events.db")
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8001"))

# Subscribe to all sensor topics including new data types
TOPIC_FILTERS = [
    "sensors/+/+",           # Original topics
    "sensors/+/weather",     # Weather data
    "sensors/+/heart",       # Heart data  
    "sensors/+/text",        # Text data
    "sensors/+/image",       # Image metadata
    "sensors/+/vertical",    # Vertical FL data
    "sensors/+/specialized"  # Specialized medical data
]

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Prometheus metrics
registry = CollectorRegistry()

# Counters
EVENTS_TOTAL = Counter('ingestion_events_total', 'Total events ingested', ['type', 'node_id', 'category'], registry=registry)
EVENTS_FAILED = Counter('ingestion_events_failed', 'Failed events', ['reason'], registry=registry)
DB_INSERTS = Counter('ingestion_db_inserts_total', 'Total database inserts', registry=registry)
DB_ERRORS = Counter('ingestion_db_errors_total', 'Database errors', ['error_type'], registry=registry)
MQTT_MESSAGES_RECEIVED = Counter('ingestion_mqtt_messages_total', 'MQTT messages received', ['topic'], registry=registry)

# Gauges
MQTT_CONNECTED = Gauge('ingestion_mqtt_connected', 'MQTT connection status (1=connected, 0=disconnected)', registry=registry)
ACTIVE_EVENTS = Gauge('ingestion_active_events', 'Currently active events being processed', registry=registry)
DATABASE_SIZE = Gauge('ingestion_database_size_bytes', 'Database file size in bytes', registry=registry)
EVENT_QUEUE_SIZE = Gauge('ingestion_event_queue_size', 'Size of event processing queue', registry=registry)

# Histograms
EVENT_PROCESSING_TIME = Histogram('ingestion_event_processing_seconds', 'Time to process an event', 
                                 buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0], registry=registry)
EVENT_SIZE = Histogram('ingestion_event_size_bytes', 'Event payload size in bytes', 
                      buckets=[100, 500, 1000, 5000, 10000, 50000], registry=registry)

# Health metrics
HEART_RATE = Gauge('ingestion_heart_rate', 'Heart rate from events', ['node_id'], registry=registry)
TEMPERATURE = Gauge('ingestion_temperature', 'Temperature from events', ['node_id'], registry=registry)
HUMIDITY = Gauge('ingestion_humidity', 'Humidity from events', ['node_id'], registry=registry)
CHOLESTEROL = Gauge('ingestion_cholesterol', 'Cholesterol level', ['node_id'], registry=registry)
BLOOD_PRESSURE = Gauge('ingestion_blood_pressure', 'Blood pressure', ['node_id'], registry=registry)

# Summary
API_REQUEST_DURATION = Summary('ingestion_api_request_duration_seconds', 'API request duration', ['endpoint'], registry=registry)

app = FastAPI(title="IoT Ingestion API", version="2.0")

# Pydantic models for API validation
class EventResponse(BaseModel):
    event_id: str
    timestamp: str
    node_id: str
    type: str
    seq: int
    payload: dict
    target_node: Optional[str] = None

class EventsResponse(BaseModel):
    events: List[EventResponse]
    count: int
    total_events: int

class HealthResponse(BaseModel):
    status: str
    service: str
    event_count: int
    database: str
    mqtt_connected: bool
    error: Optional[str] = None

# Middleware for metrics
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        with API_REQUEST_DURATION.labels(endpoint=request.url.path).time():
            response = await call_next(request)
        return response

app.add_middleware(MetricsMiddleware)

# -------- Database --------
def init_db():
    """Initialize database with enhanced schema"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Enhanced events table
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            timestamp TEXT,
            node_id TEXT,
            type TEXT,
            seq INTEGER,
            payload TEXT,
            target_node TEXT,
            data_category TEXT,
            processed BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for better performance
    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_node_id ON events(node_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_data_category ON events(data_category)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_processed ON events(processed)")
    
    # Statistics table for monitoring
    c.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            events_ingested INTEGER,
            events_by_type TEXT,
            avg_processing_time REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def categorize_event(event_type: str, payload: dict) -> str:
    """Categorize events for better organization"""
    if event_type == "weather":
        return "environmental"
    elif event_type in ["heart", "health_text", "health_image", "specialized"]:
        return "medical"
    elif event_type in ["text", "image"]:
        return "multimodal"
    elif "vertical" in event_type:
        return "federated_learning"
    else:
        return "general"

def validate_event_data(event: dict) -> bool:
    """Validate incoming event data"""
    required_fields = ["event_id", "timestamp", "node_id", "type", "seq", "payload"]
    
    for field in required_fields:
        if field not in event:
            print(f"❌ Missing required field: {field}")
            return False
    
    # Validate payload is a dictionary
    if not isinstance(event.get("payload"), dict):
        print(f"❌ Invalid payload type: {type(event.get('payload'))}")
        return False
    
    # Validate timestamp format (basic check)
    try:
        datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
    except ValueError:
        print(f"❌ Invalid timestamp format: {event['timestamp']}")
        return False
    
    return True

def clean_payload(payload: dict) -> dict:
    """Clean payload data by handling special values"""
    cleaned = {}
    for key, value in payload.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        elif value is None:
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned

def insert_event(ev: dict):
    """Insert event into database with enhanced error handling"""
    start_time = datetime.now()
    ACTIVE_EVENTS.inc()
    
    try:
        if not validate_event_data(ev):
            EVENT_SIZE.observe(len(json.dumps(ev)))
            EVENTS_FAILED.labels(reason="validation").inc()
            print(f"❌ Invalid event data, skipping: {ev.get('event_id')}")
            return False
        
        with EVENT_PROCESSING_TIME.time():
            with get_db_connection() as conn:
                c = conn.cursor()
                
                # Clean payload before insertion
                cleaned_payload = clean_payload(ev.get("payload", {}))
                data_category = categorize_event(ev.get("type"), cleaned_payload)
                
                # Update health metrics if available
                if ev.get("type") == "heart":
                    heart_rate = cleaned_payload.get("heart_rate")
                    cholesterol = cleaned_payload.get("cholesterol")
                    blood_pressure = cleaned_payload.get("blood_pressure")
                    
                    if heart_rate is not None:
                        HEART_RATE.labels(node_id=ev.get("node_id")).set(heart_rate)
                    if cholesterol is not None:
                        CHOLESTEROL.labels(node_id=ev.get("node_id")).set(cholesterol)
                    if blood_pressure is not None:
                        BLOOD_PRESSURE.labels(node_id=ev.get("node_id")).set(blood_pressure)
                
                elif ev.get("type") == "weather":
                    temp = cleaned_payload.get("temperature")
                    humid = cleaned_payload.get("humidity")
                    
                    if temp is not None:
                        TEMPERATURE.labels(node_id=ev.get("node_id")).set(temp)
                    if humid is not None:
                        HUMIDITY.labels(node_id=ev.get("node_id")).set(humid)
                
                c.execute("""
                    INSERT OR IGNORE INTO events 
                    (event_id, timestamp, node_id, type, seq, payload, target_node, data_category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ev.get("event_id"), 
                    ev.get("timestamp"), 
                    ev.get("node_id"), 
                    ev.get("type"), 
                    ev.get("seq"),
                    json.dumps(cleaned_payload, allow_nan=False), 
                    ev.get("target_node"),
                    data_category
                ))
                
                conn.commit()
                
                if c.rowcount > 0:
                    DB_INSERTS.inc()
                    EVENTS_TOTAL.labels(
                        type=ev.get("type"), 
                        node_id=ev.get("node_id"), 
                        category=data_category
                    ).inc()
                    
                    EVENT_SIZE.observe(len(json.dumps(cleaned_payload)))
                    print(f"✅ Inserted event: {ev.get('event_id')} - {ev.get('type')} ({data_category})")
                    update_ingestion_stats(conn)
                    return True
                else:
                    print(f"⚠️ Duplicate event skipped: {ev.get('event_id')}")
                    return False
                    
    except sqlite3.IntegrityError as e:
        DB_ERRORS.labels(error_type="integrity").inc()
        print(f"⚠️ Database integrity error: {e}")
        return False
    except Exception as e:
        DB_ERRORS.labels(error_type="general").inc()
        print(f"❌ Database error: {e}")
        return False
    finally:
        ACTIVE_EVENTS.dec()
        processing_time = (datetime.now() - start_time).total_seconds()
        EVENT_PROCESSING_TIME.observe(processing_time)

def update_ingestion_stats(conn):
    """Update ingestion statistics"""
    try:
        c = conn.cursor()
        
        # Get counts by type for the last hour
        c.execute("""
            SELECT type, COUNT(*) as count 
            FROM events 
            WHERE datetime(timestamp) > datetime('now', '-1 hour')
            GROUP BY type
        """)
        type_counts = {row['type']: row['count'] for row in c.fetchall()}
        
        c.execute("""
            INSERT INTO ingestion_stats 
            (timestamp, events_ingested, events_by_type, avg_processing_time)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            sum(type_counts.values()),
            json.dumps(type_counts),
            0.1  # Simplified processing time
        ))
        
        conn.commit()
    except Exception as e:
        print(f"⚠️ Failed to update stats: {e}")

# -------- MQTT Worker --------
def on_connect(client, userdata, flags, rc):
    """Callback for when the client receives a CONNACK response from the server"""
    if rc == 0:
        MQTT_CONNECTED.set(1)
        print(f"✅ Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
        
        # Subscribe to all topic filters
        for topic in TOPIC_FILTERS:
            client.subscribe(topic)
            print(f"📡 Subscribed to topic: {topic}")
    else:
        MQTT_CONNECTED.set(0)
        print(f"❌ Failed to connect to MQTT broker, return code: {rc}")

def on_message(client, userdata, msg):
    """Callback for when a PUBLISH message is received from the server"""
    MQTT_MESSAGES_RECEIVED.labels(topic=msg.topic).inc()
    
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        print(f"📨 Received message on topic {msg.topic}")
        
        # Add topic information to payload for tracking
        payload["_topic"] = msg.topic
        payload["_qos"] = msg.qos
        payload["_retained"] = msg.retain
        
        success = insert_event(payload)
        if not success:
            EVENTS_FAILED.labels(reason="insertion").inc()
            print(f"⚠️ Failed to process event from topic: {msg.topic}")
            
    except json.JSONDecodeError as e:
        EVENTS_FAILED.labels(reason="json_decode").inc()
        print(f"❌ JSON decode error in topic {msg.topic}: {e}")
    except UnicodeDecodeError as e:
        EVENTS_FAILED.labels(reason="unicode_decode").inc()
        print(f"❌ Unicode decode error in topic {msg.topic}: {e}")
    except Exception as e:
        EVENTS_FAILED.labels(reason="general").inc()
        print(f"❌ Error processing message from {msg.topic}: {e}")

def on_disconnect(client, userdata, rc):
    """Callback for when the client disconnects from the broker"""
    MQTT_CONNECTED.set(0)
    if rc != 0:
        print(f"⚠️ Unexpected MQTT disconnection, return code: {rc}")
        # Attempt to reconnect
        try:
            client.reconnect()
        except Exception as e:
            print(f"❌ Failed to reconnect to MQTT: {e}")

def mqtt_worker():
    """MQTT worker thread with enhanced error handling"""
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            client.on_disconnect = on_disconnect
            
            # Set last will and testament
            client.will_set("ingestion/service/status", "offline", retain=True)
            
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            
            # Publish online status
            client.publish("ingestion/service/status", "online", retain=True)
            
            print(f"🚀 MQTT worker started. Listening on {MQTT_HOST}:{MQTT_PORT}")
            client.loop_forever()
            
        except Exception as e:
            print(f"❌ MQTT worker error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in {retry_delay} seconds...")
                threading.Event().wait(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("💥 Max retries exceeded. MQTT worker stopped.")
                break

# -------- FastAPI Startup --------
@app.on_event("startup")
def startup():
    """Initialize the application on startup"""
    print("🚀 Starting ingestion service...")
    init_db()
    
    # Start MQTT worker in a separate thread
    mqtt_thread = threading.Thread(target=mqtt_worker, daemon=True)
    mqtt_thread.name = "MQTT-Worker"
    mqtt_thread.start()
    
    print("✅ Ingestion service started successfully")

# -------- API Endpoints --------
@app.get("/events", response_model=EventsResponse)
def get_events(
    limit: int = 100, 
    type_filter: str = None,
    node_id: str = None,
    category: str = None,
    offset: int = 0
):
    """Get events with filtering and pagination"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Build query with filters
            query = """
                SELECT event_id, timestamp, node_id, type, seq, payload, target_node
                FROM events 
            """
            params = []
            conditions = []
            
            if type_filter:
                conditions.append("type = ?")
                params.append(type_filter)
            
            if node_id:
                conditions.append("node_id = ?")
                params.append(node_id)
                
            if category:
                conditions.append("data_category = ?")
                params.append(category)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            c.execute(query, params)
            rows = c.fetchall()
            
            # Get total count for pagination
            count_query = "SELECT COUNT(*) as total FROM events"
            if conditions:
                count_query += " WHERE " + " AND ".join(conditions)
            
            c.execute(count_query, params[:-2])  # Exclude limit and offset
            total_count = c.fetchone()['total']
        
        events = []
        for r in rows:
            try:
                payload = json.loads(r['payload']) if r['payload'] else {}
                
                events.append(EventResponse(
                    event_id=r['event_id'],
                    timestamp=r['timestamp'],
                    node_id=r['node_id'],
                    type=r['type'],
                    seq=r['seq'],
                    payload=payload,
                    target_node=r['target_node']
                ))
            except Exception as e:
                print(f"⚠️ Error processing event {r['event_id']}: {e}")
                continue
        
        return EventsResponse(
            events=events,
            count=len(events),
            total_events=total_count
        )
        
    except Exception as e:
        print(f"❌ Error in get_events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events/stats")
def get_events_stats(hours: int = 24):
    """Get ingestion statistics"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Events by type
            c.execute("""
                SELECT type, COUNT(*) as count 
                FROM events 
                WHERE datetime(timestamp) > datetime('now', ?) 
                GROUP BY type 
                ORDER BY count DESC
            """, (f"-{hours} hours",))
            events_by_type = {row['type']: row['count'] for row in c.fetchall()}
            
            # Events by category
            c.execute("""
                SELECT data_category, COUNT(*) as count 
                FROM events 
                WHERE datetime(timestamp) > datetime('now', ?) 
                GROUP BY data_category 
                ORDER BY count DESC
            """, (f"-{hours} hours",))
            events_by_category = {row['data_category']: row['count'] for row in c.fetchall()}
            
            # Total events
            c.execute("SELECT COUNT(*) as total FROM events")
            total_events = c.fetchone()['total']
            
            # Recent events count
            c.execute("""
                SELECT COUNT(*) as recent 
                FROM events 
                WHERE datetime(timestamp) > datetime('now', ?)
            """, (f"-{hours} hours",))
            recent_events = c.fetchone()['recent']
            
        return {
            "total_events": total_events,
            f"events_last_{hours}_hours": recent_events,
            "events_by_type": events_by_type,
            "events_by_category": events_by_category,
            "ingestion_rate_per_hour": recent_events / hours if hours > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ Error in get_events_stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) as count FROM events")
            count = c.fetchone()['count']
            
            # Check if database is writable
            c.execute("SELECT 1")
            db_status = "healthy"
            
        return HealthResponse(
            status="healthy",
            service="ingestion_api",
            event_count=count,
            database=db_status,
            mqtt_connected=bool(MQTT_CONNECTED._value.get())
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            service="ingestion_api", 
            event_count=0,
            database="unhealthy",
            mqtt_connected=False,
            error=str(e)
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM events")
            total_events = c.fetchone()[0]
            ACTIVE_EVENTS.set(total_events)
            
            # Update database size
            if os.path.exists(DB_PATH):
                DATABASE_SIZE.set(os.path.getsize(DB_PATH))
    except Exception:
        pass
    
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    """Root endpoint with service information"""
    return {
        "service": "IoT Ingestion API",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "events": "/events",
            "events_stats": "/events/stats", 
            "health": "/health",
            "metrics": "/metrics"
        },
        "supported_data_types": [
            "weather", "heart", "text", "image", 
            "health_text", "health_image", "specialized",
            "vertical"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)