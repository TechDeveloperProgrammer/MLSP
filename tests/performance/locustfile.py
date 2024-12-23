from locust import HttpUser, task, between
import random

class MLSPPerformanceTest(HttpUser):
    """
    Performance testing for MLSP-MACP platform
    
    Simulates various user interactions and load scenarios
    """
    
    wait_time = between(1, 5)  # Random wait between 1-5 seconds
    
    def __init__(self, *args, **kwargs):
        """Initialize performance test user"""
        super().__init__(*args, **kwargs)
        self.world_generation_seeds = [
            random.randint(1, 1000000) for _ in range(10)
        ]
    
    @task(3)  # Higher weight for world generation
    def test_world_generation(self):
        """Simulate world generation request"""
        seed = random.choice(self.world_generation_seeds)
        payload = {
            "seed": seed,
            "world_size": [512, 512],
            "complexity": "advanced"
        }
        
        with self.client.post(
            "/api/world/generate", 
            json=payload, 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"World generation failed: {response.text}")
    
    @task(2)  # Moderate weight for mod generation
    def test_mod_generation(self):
        """Simulate AI-driven mod generation"""
        payload = {
            "mod_type": random.choice([
                "forge", "fabric", "bukkit"
            ]),
            "minecraft_version": "1.18.2",
            "features": [
                "Custom Items", 
                "New Biomes", 
                "Performance Optimization"
            ]
        }
        
        with self.client.post(
            "/api/mod/generate", 
            json=payload, 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Mod generation failed: {response.text}")
    
    @task(1)  # Lower weight for content search
    def test_content_search(self):
        """Simulate mod/plugin content search"""
        query_terms = [
            "optimization", 
            "world generation", 
            "new biomes", 
            "performance"
        ]
        
        payload = {
            "query": random.choice(query_terms),
            "minecraft_version": "1.18.2",
            "sources": ["modrinth", "curseforge"]
        }
        
        with self.client.get(
            "/api/content/search", 
            params=payload, 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Content search failed: {response.text}")
    
    @task(1)  # Performance health check
    def test_platform_health(self):
        """Check platform health endpoint"""
        with self.client.get(
            "/health", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Platform health check failed: {response.text}")

def main():
    """
    Run performance tests
    
    Simulates various load scenarios for MLSP-MACP platform
    """
    import os
    
    # Locust configuration
    os.system(
        "locust -f tests/performance/locustfile.py "
        "--headless "
        "-u 100 "    # 100 concurrent users
        "-r 10 "     # 10 users spawned per second
        "-t 1h "     # Run for 1 hour
    )

if __name__ == '__main__':
    main()
