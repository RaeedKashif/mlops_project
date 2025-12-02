import os
import json
import uuid
import asyncio
import random
from datetime import datetime, timezone

import pandas as pd
import paho.mqtt.client as mqtt

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
RATE = float(os.getenv("RATE", "2"))

# -------- Node Definitions -------- #
CONTINENTS = ["asia", "europe", "africa", "north_america", "south_america", "australia"]
HOSPITALS = ["A", "B", "C"]

# -------- Helpers -------- #
def iso_now():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def make_msg(node, dtype, seq, payload, target_node=None):
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": iso_now(),
        "node_id": node,
        "type": dtype,
        "seq": seq,
        "payload": payload,
        "target_node": target_node
    }

def build_client():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT)
    client.loop_start()
    return client

def validate_payload(payload):
    """Validate and clean payload data"""
    cleaned = {}
    for key, value in payload.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Ensure numeric values are within reasonable bounds
            if "rate" in key.lower() and (value < 0 or value > 300):
                value = max(0, min(value, 300))
            elif "temperature" in key.lower() and (value < -100 or value > 200):
                value = max(-100, min(value, 200))
        cleaned[key] = value
    return cleaned

# -------- Publisher Loops -------- #
async def publish_weather(client, df):
    seq = 0
    if df.empty:
        print("Warning: No weather data available")
        return
        
    while True:
        for _, row in df.iterrows():
            # Extract weather data with proper fallbacks
            temperature = row.get("temperatureMax") or row.get("temperatureHigh") or row.get("temperature")
            humidity = row.get("humidity")
            
            if temperature is None or pd.isna(temperature) or humidity is None or pd.isna(humidity):
                continue
                
            payload = validate_payload({
                "temperature": float(temperature),
                "humidity": float(humidity),
                "pressure": random.uniform(980, 1040)  # Add some variety
            })
            
            continent = random.choice(CONTINENTS)
            node = f"continent_{continent}"
            target_node = f"continent_{random.choice([c for c in CONTINENTS if c != continent])}"
            
            msg = make_msg(node, "weather", seq, payload, target_node)
            topic = f"sensors/{node}/weather"
            client.publish(topic, json.dumps(msg))
            print(f"🌤 Weather: {payload} from {node} to {target_node}")
            seq += 1
            await asyncio.sleep(RATE)
        
        # Shuffle data for variety
        df = df.sample(frac=1).reset_index(drop=True)

async def publish_heart(client, df):
    seq = 0
    if df.empty:
        print("Warning: No heart data available")
        return
        
    while True:
        for _, row in df.iterrows():
            # Use available heart-related columns
            heart_rate = row.get("thalachh")  # Max heart rate achieved
            cholesterol = row.get("chol")     # Cholesterol level
            blood_pressure = row.get("trtbps") # Resting blood pressure
            
            payload = validate_payload({
                "heart_rate": float(heart_rate) if pd.notna(heart_rate) else random.randint(60, 180),
                "cholesterol": float(cholesterol) if pd.notna(cholesterol) else random.randint(100, 400),
                "blood_pressure": float(blood_pressure) if pd.notna(blood_pressure) else random.randint(90, 200),
                "patient_age": random.randint(25, 80)  # Additional context
            })
            
            hospital = random.choice(HOSPITALS)
            node = f"hospital_{hospital}"
            target_node = f"hospital_{random.choice([h for h in HOSPITALS if h != hospital])}"
            
            msg = make_msg(node, "heart", seq, payload, target_node)
            topic = f"sensors/{node}/heart"
            client.publish(topic, json.dumps(msg))
            print(f"❤️ Heart: HR={payload['heart_rate']} from {node} to {target_node}")
            seq += 1
            await asyncio.sleep(RATE)
        
        df = df.sample(frac=1).reset_index(drop=True)

async def publish_text_data(client, heart_df):
    """Simulate text data (medical notes, reports) with context from heart data"""
    seq = 0
    medical_notes = [
        "Patient shows improved cardiac function with regular medication",
        "Elevated cholesterol levels observed, recommend dietary changes",
        "Blood pressure within normal range, continue current treatment",
        "Mild tachycardia detected, schedule follow-up appointment",
        "Excellent response to treatment plan, vital signs stable",
        "Moderate hypertension noted, adjust medication dosage",
        "Cholesterol levels improved with current statin therapy",
        "Cardiac stress test results within normal limits"
    ]
    
    while True:
        for hospital in HOSPITALS:
            # Use heart data context for more realistic notes
            if not heart_df.empty:
                heart_context = heart_df.sample(1).iloc[0]
                hr = heart_context.get('thalachh', random.randint(60, 180))
                chol = heart_context.get('chol', random.randint(100, 400))
                
                # Select note based on context
                if hr > 160:
                    note = "Patient exhibits elevated heart rate during stress test"
                elif chol > 240:
                    note = "High cholesterol levels require dietary intervention"
                else:
                    note = random.choice(medical_notes)
            else:
                note = random.choice(medical_notes)
            
            payload = validate_payload({
                "medical_note": note,
                "sentiment_score": random.uniform(0.1, 0.9),
                "urgency_level": random.choice(["low", "medium", "high"]),
                "note_length": len(note),
                "confidence_score": random.uniform(0.7, 0.95)
            })
            
            node = f"hospital_{hospital}"
            msg = make_msg(node, "text", seq, payload)
            topic = f"sensors/{node}/text"
            client.publish(topic, json.dumps(msg))
            print(f"📝 Text: '{note[:40]}...' from {node}")
            seq += 1
            await asyncio.sleep(RATE * 2)  # Slower rate for text data

async def publish_image_metadata(client):
    """Simulate image data metadata (X-rays, MRI, etc.)"""
    seq = 0
    image_types = ["xray_chest", "mri_brain", "ct_scan", "ultrasound", "pet_scan"]
    findings_map = {
        "xray_chest": ["clear", "pneumonia", "fibrosis", "nodule", "cardiomegaly"],
        "mri_brain": ["normal", "tumor", "stroke", "ms_lesions", "atrophy"],
        "ct_scan": ["clear", "fracture", "hemorrhage", "mass", "infection"],
        "ultrasound": ["normal", "gallstones", "cyst", "mass", "inflammation"],
        "pet_scan": ["normal", "metastasis", "inflammation", "infection", "tumor"]
    }
    
    while True:
        for hospital in HOSPITALS:
            image_type = random.choice(image_types)
            possible_findings = findings_map.get(image_type, ["normal", "abnormal", "inconclusive"])
            
            payload = validate_payload({
                "image_type": image_type,
                "image_size": random.randint(1000000, 5000000),
                "resolution": f"{random.randint(1000, 4000)}x{random.randint(1000, 4000)}",
                "quality_score": random.uniform(0.7, 0.99),
                "findings": random.choice(possible_findings),
                "contrast_used": random.choice([True, False]),
                "radiology_score": random.uniform(0.5, 1.0)
            })
            
            node = f"hospital_{hospital}"
            msg = make_msg(node, "image", seq, payload)
            topic = f"sensors/{node}/image"
            client.publish(topic, json.dumps(msg))
            print(f"🖼 Image: {image_type} ({payload['findings']}) from {node}")
            seq += 1
            await asyncio.sleep(RATE * 3)  # Slowest rate for image data

async def publish_vertical_features(client, weather_df, heart_df):
    """Publish vertically partitioned features for hybrid FL approach"""
    seq = 0
    
    while True:
        # Hospital A: Health + Text features (comprehensive patient profile)
        if seq % 4 == 0 and not heart_df.empty:
            health_data = heart_df.sample(1).iloc[0]
            payload = validate_payload({
                # Health metrics
                "heart_rate": float(health_data.get('thalachh', 72)),
                "cholesterol": float(health_data.get('chol', 200)),
                "blood_pressure": float(health_data.get('trtbps', 120)),
                # Text features
                "medical_note": "Comprehensive patient assessment completed",
                "note_quality_score": 0.92,
                "assessment_type": "routine_checkup"
            })
            msg = make_msg("hospital_A", "health_text", seq, payload)
            client.publish("sensors/hospital_A/vertical", json.dumps(msg))
            print(f"🔄 Vertical: Health+Text from hospital_A")
        
        # Hospital B: Health + Image metadata (diagnostic focus)
        elif seq % 4 == 1 and not heart_df.empty:
            health_data = heart_df.sample(1).iloc[0]
            payload = validate_payload({
                # Health metrics
                "heart_rate": float(health_data.get('thalachh', 72)),
                "cholesterol": float(health_data.get('chol', 200)),
                "blood_pressure": float(health_data.get('trtbps', 120)),
                # Image features
                "image_type": random.choice(["xray_chest", "ct_scan"]),
                "image_quality": 0.94,
                "findings_confidence": 0.87,
                "diagnostic_priority": random.choice(["routine", "urgent"])
            })
            msg = make_msg("hospital_B", "health_image", seq, payload)
            client.publish("sensors/hospital_B/vertical", json.dumps(msg))
            print(f"🔄 Vertical: Health+Image from hospital_B")
        
        # Hospital C: Specialized metrics
        elif seq % 4 == 2 and not heart_df.empty:
            health_data = heart_df.sample(1).iloc[0]
            payload = validate_payload({
                "ecg_reading": random.uniform(-0.5, 2.0),
                "oxygen_saturation": random.uniform(95.0, 99.9),
                "respiratory_rate": random.randint(12, 20),
                "blood_glucose": random.uniform(70, 140)
            })
            msg = make_msg("hospital_C", "specialized", seq, payload)
            client.publish("sensors/hospital_C/vertical", json.dumps(msg))
            print(f"🔄 Vertical: Specialized from hospital_C")
        
        # Continents: Weather + Environmental data
        elif seq % 4 == 3 and not weather_df.empty:
            weather_data = weather_df.sample(1).iloc[0]
            payload = validate_payload({
                "temperature": float(weather_data.get('temperatureMax', 70)),
                "humidity": float(weather_data.get('humidity', 50)),
                "pressure": 1013.25 + random.uniform(-20, 20),
                "air_quality": random.choice(["good", "moderate", "poor"]),
                "uv_index": random.randint(1, 11),
                "wind_speed": random.uniform(0, 25)
            })
            continent = random.choice(CONTINENTS)
            msg = make_msg(f"continent_{continent}", "weather_env", seq, payload)
            client.publish(f"sensors/continent_{continent}/vertical", json.dumps(msg))
            print(f"🔄 Vertical: Weather+Env from continent_{continent}")
        
        seq += 1
        await asyncio.sleep(RATE)

# -------- Main Loop -------- #
async def main():
    # Load datasets
    try:
        weather = pd.read_csv("data/weather.csv")
        print(f"✅ Weather data loaded: {len(weather)} records")
    except Exception as e:
        print(f"❌ Error loading weather data: {e}")
        weather = pd.DataFrame()
    
    try:
        heart = pd.read_csv("data/heart.csv")
        print(f"✅ Heart data loaded: {len(heart)} records")
    except Exception as e:
        print(f"❌ Error loading heart data: {e}")
        heart = pd.DataFrame()
    
    # Build MQTT client
    client = build_client()
    print(f"✅ MQTT client connected to {MQTT_HOST}:{MQTT_PORT}")
    
    # Start all publishers
    publishers = [
        publish_weather(client, weather.copy()),
        publish_heart(client, heart.copy()),
        publish_text_data(client, heart.copy() if not heart.empty else pd.DataFrame()),
        publish_image_metadata(client),
        publish_vertical_features(client, weather.copy(), heart.copy())
    ]
    
    print("🚀 Starting all MQTT publishers...")
    await asyncio.gather(*publishers)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 MQTT publisher stopped by user")
    except Exception as e:
        print(f"💥 Critical error: {e}")