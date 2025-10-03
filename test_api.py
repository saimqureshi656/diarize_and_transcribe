#!/usr/bin/env python3
"""
Simple test script to verify the API is working
"""
import requests
import sys
import json

def test_health(base_url):
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   GPU Available: {data['gpu_available']}")
            print(f"   GPU Name: {data['gpu_name']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_process(base_url, audio_file):
    """Test process endpoint"""
    print(f"\n🔍 Testing process endpoint with {audio_file}...")
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/process", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Processing successful!")
            print(f"   Job ID: {data['job_id']}")
            print(f"   Beep Duration: {data['beep_duration_seconds']}s")
            print(f"   Total Segments: {data['total_segments']}")
            print("\n📝 Results:")
            for i, result in enumerate(data['results'][:3]):  # Show first 3
                print(f"   [{i+1}] {result['speaker']}: {result['text'][:50]}...")
            
            # Save full results
            output_file = f"test_results_{data['job_id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Full results saved to: {output_file}")
            return True
        else:
            print(f"❌ Processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Configuration
    BASE_URL = "http://localhost:8001"  # Change if testing remotely
    
    print("="*60)
    print("🧪 API Testing Script")
    print("="*60)
    
    # Test health
    if not test_health(BASE_URL):
        print("\n❌ Health check failed. Make sure the API is running.")
        sys.exit(1)
    
    # Test process if audio file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        test_process(BASE_URL, audio_file)
    else:
        print("\n💡 To test processing, run:")
        print(f"   python3 test_api.py /path/to/audio.wav")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)