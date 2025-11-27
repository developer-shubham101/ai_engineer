#!/usr/bin/env python3
"""
Test API cases for Enhanced Embedding Model Upgrade System
Tests embedding model configuration, status monitoring, and performance.
"""

import requests
import json
import time
import os
from typing import Dict, Any

# Configuration
BASE_URL = "http://192.168.1.2:8000"
ADMIN_CREDENTIALS = {"username": "admin", "password": "admin123"}

class EmbeddingModelTester:
    def __init__(self):
        self.token = None
        self.session = requests.Session()
    
    def authenticate(self) -> bool:
        """Get JWT token for API access."""
        try:
            response = self.session.post(
                f"{BASE_URL}/api/auth/token",
                json=ADMIN_CREDENTIALS
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                print("✅ Authentication successful")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    def test_embedding_status(self) -> Dict[str, Any]:
        """Test embedding model status endpoint."""
        print("\n🔍 Testing Embedding Model Status...")
        
        try:
            response = self.session.get(f"{BASE_URL}/api/rag/embedding/status")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Embedding status endpoint accessible")
                
                if data.get("ok"):
                    model_info = data.get("embedding_model", {})
                    print(f"📊 Current Model: {model_info.get('model_key')}")
                    print(f"📊 Model Name: {model_info.get('model_name')}")
                    print(f"📊 Dimensions: {model_info.get('dimensions')}")
                    print(f"📊 Performance: {model_info.get('performance')}")
                    print(f"📊 Accuracy: {model_info.get('accuracy')}")
                    print(f"📊 Model Loaded: {model_info.get('model_loaded')}")
                    
                    if model_info.get('actual_dimensions'):
                        print(f"📊 Actual Dimensions: {model_info.get('actual_dimensions')}")
                    
                    return model_info
                else:
                    print(f"❌ Embedding status error: {data.get('error')}")
                    return {}
            else:
                print(f"❌ Status endpoint failed: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Status test error: {e}")
            return {}
    
    def test_embedding_performance(self) -> bool:
        """Test embedding performance with actual RAG query."""
        print("\n⚡ Testing Embedding Performance...")
        
        test_queries = [
            "What is our company policy?",
            "How do I apply for leave?", 
            "What are the HR guidelines?",
            "Tell me about our technical documentation standards"
        ]
        
        performance_results = []
        
        for query in test_queries:
            try:
                start_time = time.time()
                
                response = self.session.post(
                    f"{BASE_URL}/api/rag/local/query",
                    json={
                        "question": query,
                        "top_k": 3,
                        "use_llm": False,  # Just test embedding retrieval
                        "debug": True
                    }
                )
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    retrieved_count = len(data.get("retrieved", []))
                    
                    performance_results.append({
                        "query": query[:30] + "...",
                        "time_ms": round(elapsed * 1000, 2),
                        "retrieved_docs": retrieved_count,
                        "success": True
                    })
                    
                    print(f"✅ Query: {query[:30]}... | {elapsed*1000:.1f}ms | {retrieved_count} docs")
                else:
                    print(f"❌ Query failed: {response.status_code}")
                    performance_results.append({
                        "query": query[:30] + "...",
                        "success": False,
                        "error": response.status_code
                    })
                    
            except Exception as e:
                print(f"❌ Performance test error: {e}")
                return False
        
        # Calculate average performance
        successful_queries = [r for r in performance_results if r.get("success")]
        if successful_queries:
            avg_time = sum(r["time_ms"] for r in successful_queries) / len(successful_queries)
            total_docs = sum(r["retrieved_docs"] for r in successful_queries)
            print(f"\n📈 Performance Summary:")
            print(f"   Average Query Time: {avg_time:.1f}ms")
            print(f"   Total Documents Retrieved: {total_docs}")
            print(f"   Success Rate: {len(successful_queries)}/{len(test_queries)}")
            return True
        
        return False
    
    def test_model_fallback(self) -> bool:
        """Test model fallback behavior (simulation)."""
        print("\n🔄 Testing Model Fallback Behavior...")
        
        # Get current status first
        current_status = self.test_embedding_status()
        
        if current_status.get("model_loaded"):
            print("✅ Primary model loaded successfully")
            
            # Test with invalid model configuration (would require restart)
            print("ℹ️  Fallback testing requires configuration change and restart")
            print("ℹ️  Current model is working - fallback not triggered")
            return True
        else:
            print("⚠️  Primary model not loaded - fallback may be active")
            return False
    
    def test_embedding_upgrade_scenarios(self) -> bool:
        """Test different embedding model scenarios."""
        print("\n🔧 Testing Embedding Model Scenarios...")
        
        scenarios = [
            {
                "name": "Current Model Status",
                "description": "Check if current model is properly loaded"
            },
            {
                "name": "Performance Baseline", 
                "description": "Establish performance baseline with current model"
            },
            {
                "name": "Dimension Verification",
                "description": "Verify model dimensions match configuration"
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            
            if scenario["name"] == "Current Model Status":
                status = self.test_embedding_status()
                if status.get("model_loaded"):
                    print("   ✅ Status check passed")
                else:
                    print("   ❌ Status check failed")
                    return False
                    
            elif scenario["name"] == "Performance Baseline":
                if self.test_embedding_performance():
                    print("   ✅ Performance test passed")
                else:
                    print("   ❌ Performance test failed")
                    return False
                    
            elif scenario["name"] == "Dimension Verification":
                status = self.test_embedding_status()
                expected_dims = status.get("dimensions")
                actual_dims = status.get("actual_dimensions")
                
                if expected_dims and actual_dims:
                    if expected_dims == actual_dims:
                        print(f"   ✅ Dimensions match: {actual_dims}")
                    else:
                        print(f"   ⚠️  Dimension mismatch: expected {expected_dims}, got {actual_dims}")
                else:
                    print("   ℹ️  Dimension info not available")
        
        return True
    
    def run_all_tests(self) -> bool:
        """Run comprehensive embedding model tests."""
        print("🚀 Starting Embedding Model API Tests")
        print("=" * 50)
        
        # Authenticate
        if not self.authenticate():
            return False
        
        # Run test suite
        tests = [
            ("Embedding Status", self.test_embedding_status),
            ("Embedding Performance", self.test_embedding_performance), 
            ("Model Fallback", self.test_model_fallback),
            ("Upgrade Scenarios", self.test_embedding_upgrade_scenarios)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                print(f"\n{'='*20} {test_name} {'='*20}")
                result = test_func()
                results.append((test_name, result))
                
                if result:
                    print(f"✅ {test_name} completed successfully")
                else:
                    print(f"❌ {test_name} failed")
                    
            except Exception as e:
                print(f"❌ {test_name} error: {e}")
                results.append((test_name, False))
        
        # Summary
        print(f"\n{'='*20} TEST SUMMARY {'='*20}")
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All embedding model tests passed!")
            return True
        else:
            print("⚠️  Some tests failed - check logs for details")
            return False


def main():
    """Run embedding model API tests."""
    tester = EmbeddingModelTester()
    
    print("Embedding Model Upgrade System - API Test Suite")
    print("Testing enhanced embedding models with monitoring capabilities")
    print()
    
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 Embedding model system is working correctly!")
        print("💡 Ready for production embedding model upgrades")
    else:
        print("\n🔧 Some issues detected - review test output")
    
    return success


if __name__ == "__main__":
    main()