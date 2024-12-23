import os
import json
import logging
import sys
from datetime import datetime, timedelta
import tweepy
import requests

class MLSPSocialMediaBot:
    """
    Social Media Bot for MLSP Project Updates and Engagement
    
    Features:
    - Automated project update tweets
    - GitHub release announcements
    - Community engagement tracking
    - Trending topic monitoring
    """
    
    def __init__(self, config_path='../config/social_media_config.json'):
        """
        Initialize Social Media Bot
        
        :param config_path: Path to configuration file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Twitter API client
        self.twitter_client = self.initialize_twitter_client()
    
    def load_config(self, config_path):
        """
        Load bot configuration
        
        :param config_path: Path to configuration file
        :return: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in configuration file: {config_path}")
            return {}
    
    def setup_logging(self):
        """
        Configure logging for the bot
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='mlsp_social_media_bot.log'
        )
        
        self.logger = logging.getLogger('MLSPSocialMediaBot')
    
    def initialize_twitter_client(self):
        """
        Initialize Twitter API client
        
        :return: Authenticated Twitter API client
        """
        try:
            client = tweepy.Client(
                bearer_token=self.config.get('TWITTER_BEARER_TOKEN'),
                consumer_key=self.config.get('TWITTER_CONSUMER_KEY'),
                consumer_secret=self.config.get('TWITTER_CONSUMER_SECRET'),
                access_token=self.config.get('TWITTER_ACCESS_TOKEN'),
                access_token_secret=self.config.get('TWITTER_ACCESS_TOKEN_SECRET')
            )
            return client
        except Exception as e:
            self.logger.error(f"Twitter client initialization error: {e}")
            return None
    
    def fetch_github_releases(self):
        """
        Fetch latest GitHub releases
        
        :return: List of recent releases
        """
        try:
            repo = self.config.get('GITHUB_REPO', 'yourusername/MLSP')
            url = f'https://api.github.com/repos/{repo}/releases'
            
            headers = {
                'Authorization': f"token {self.config.get('GITHUB_TOKEN')}"
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"GitHub releases fetch error: {response.status_code}")
                return []
        
        except Exception as e:
            self.logger.error(f"GitHub releases fetch error: {e}")
            return []
    
    def generate_release_tweet(self, release):
        """
        Generate tweet for GitHub release
        
        :param release: Release metadata
        :return: Formatted tweet text
        """
        version = release.get('tag_name', 'Unknown Version')
        description = release.get('body', '')
        
        # Truncate description if too long
        description = description[:200] + '...' if len(description) > 200 else description
        
        tweet = f"""🚀 New MLSP Release: {version}

{description}

Full details: {release.get('html_url')}

#MinecraftMod #OpenSource #GameDev"""
        
        return tweet
    
    def post_release_tweet(self, release):
        """
        Post tweet about GitHub release
        
        :param release: Release metadata
        """
        try:
            tweet_text = self.generate_release_tweet(release)
            
            if self.twitter_client:
                response = self.twitter_client.create_tweet(text=tweet_text)
                self.logger.info(f"Released tweet for version {release.get('tag_name')}")
            else:
                self.logger.error("Twitter client not initialized")
        
        except Exception as e:
            self.logger.error(f"Tweet posting error: {e}")
    
    def monitor_trending_topics(self):
        """
        Monitor trending topics related to Minecraft and game development
        """
        try:
            # Placeholder for more advanced trending topic tracking
            trending_keywords = [
                'Minecraft', 'GameDev', 'OpenSource', 
                'AIInGaming', 'ProceduralGeneration'
            ]
            
            for keyword in trending_keywords:
                tweets = self.twitter_client.search_recent_tweets(
                    query=keyword, 
                    max_results=10
                )
                
                # Log or process trending tweets
                self.logger.info(f"Trending tweets for {keyword}: {len(tweets.data)}")
        
        except Exception as e:
            self.logger.error(f"Trending topics monitoring error: {e}")
    
    def generate_community_quest_tweet(self):
        """
        Generate and tweet a sample quest for community engagement
        
        :return: Generated quest tweet
        """
        try:
            # Import Quest Generator
            sys.path.append('/home/veronicae/CascadeProjects/MLSP/modules/ai-generation/npc')
            from quest_generator import QuestGenerator
            
            # Create quest generator
            quest_gen = QuestGenerator()
            
            # Sample NPC profile
            npc_profile = {
                'name': 'Community Quest Master',
                'id': 'twitter_community_npc',
                'personality_traits': ['helpful', 'adventurous', 'curious']
            }
            
            # Generate quest
            quest = quest_gen.generate_quest(npc_profile)
            
            # Create tweet text
            tweet = f"""🎮 MLSP Community Quest Alert! 🚀

{quest['title']}

Difficulty: {quest['difficulty'].upper()}
Type: {quest['type'].capitalize()}

Objectives:
{' | '.join(quest['objectives'][:2])}

#MinecraftQuest #MLSPCommunity #GameDev"""
            
            return tweet
        
        except Exception as e:
            self.logger.error(f"Quest tweet generation error: {e}")
            return None
    
    def post_community_quest(self):
        """
        Post a community quest tweet
        """
        try:
            quest_tweet = self.generate_community_quest_tweet()
            
            if quest_tweet and self.twitter_client:
                response = self.twitter_client.create_tweet(text=quest_tweet)
                self.logger.info("Posted community quest tweet")
        
        except Exception as e:
            self.logger.error(f"Community quest tweet posting error: {e}")
    
    def run(self):
        """
        Run social media bot tasks
        """
        # Fetch and post latest releases
        releases = self.fetch_github_releases()
        
        for release in releases[:3]:  # Post latest 3 releases
            release_date = datetime.strptime(
                release.get('published_at'), 
                "%Y-%m-%dT%H:%M:%SZ"
            )
            
            # Only post releases from last 7 days
            if datetime.utcnow() - release_date < timedelta(days=7):
                self.post_release_tweet(release)
        
        # Monitor trending topics
        self.monitor_trending_topics()
        
        # Post a community quest periodically
        self.post_community_quest()

def main():
    """
    Initialize and run the MLSP Social Media Bot
    """
    bot = MLSPSocialMediaBot()
    bot.run()

if __name__ == '__main__':
    main()
