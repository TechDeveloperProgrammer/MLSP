import os
import json
import logging
from datetime import datetime, timedelta
import requests
from github import Github, GithubObject
import sys

class GitHubDiscussionsManager:
    """
    GitHub Discussions Management System for MLSP
    
    Features:
    - Automated discussion creation
    - Community engagement tracking
    - Discussion categorization
    - Moderation support
    """
    
    def __init__(self, config_path='../config/github_discussions_config.json'):
        """
        Initialize GitHub Discussions Manager
        
        :param config_path: Path to configuration file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize GitHub client
        self.github_client = self.initialize_github_client()
    
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
        Configure logging for the manager
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='mlsp_github_discussions.log'
        )
        
        self.logger = logging.getLogger('GitHubDiscussionsManager')
    
    def initialize_github_client(self):
        """
        Initialize GitHub API client
        
        :return: Authenticated GitHub client
        """
        try:
            client = Github(self.config.get('GITHUB_TOKEN'))
            return client
        except Exception as e:
            self.logger.error(f"GitHub client initialization error: {e}")
            return None
    
    def create_discussion_categories(self, repo):
        """
        Create discussion categories if they don't exist
        
        :param repo: GitHub repository
        """
        categories = [
            {
                'name': 'Ideas',
                'description': 'Suggest new features or improvements',
                'color': 'green'
            },
            {
                'name': 'Q&A',
                'description': 'Ask questions about the project',
                'color': 'blue'
            },
            {
                'name': 'Development',
                'description': 'Technical discussions about the project',
                'color': 'purple'
            },
            {
                'name': 'Community',
                'description': 'General community discussions',
                'color': 'yellow'
            }
        ]
        
        for category in categories:
            try:
                repo.create_discussion_category(
                    category['name'], 
                    category['description']
                )
                self.logger.info(f"Created discussion category: {category['name']}")
            except Exception as e:
                self.logger.warning(f"Category {category['name']} might already exist: {e}")
    
    def create_welcome_discussion(self, repo):
        """
        Create a welcome discussion for new community members
        
        :param repo: GitHub repository
        """
        welcome_title = "Welcome to the MLSP Community! 🎉"
        welcome_body = """
## Welcome to the Minecraft Local Server Platform Community! 

We're excited to have you here. This is a space for:
- Sharing ideas
- Asking questions
- Collaborating on development
- Building an amazing Minecraft modding experience

### Getting Started
1. Read our [Contributing Guidelines](CONTRIBUTING.md)
2. Check out our [README](README.md)
3. Explore our [Project Structure](PROJECT_STRUCTURE.md)

### How to Engage
- Use appropriate discussion categories
- Be respectful and constructive
- Share your creativity and insights

Looking forward to building something amazing together! 🚀🌍
"""
        
        try:
            repo.create_discussion(
                title=welcome_title,
                body=welcome_body,
                category='Community'
            )
            self.logger.info("Created welcome discussion")
        except Exception as e:
            self.logger.error(f"Welcome discussion creation error: {e}")
    
    def create_monthly_discussion_topics(self, repo):
        """
        Create monthly discussion topics
        
        :param repo: GitHub repository
        """
        monthly_topics = [
            {
                'title': 'Feature Wishlist: December 2024',
                'body': '## What features would you like to see in MLSP?\n\n'
                        'Share your ideas for new functionality, improvements, '
                        'and innovations for our Minecraft server platform.',
                'category': 'Ideas'
            },
            {
                'title': 'Development Roadmap Discussion',
                'body': '## MLSP Development Roadmap\n\n'
                        'Let\'s discuss our project\'s future direction, '
                        'potential milestones, and technical challenges.',
                'category': 'Development'
            }
        ]
        
        for topic in monthly_topics:
            try:
                repo.create_discussion(
                    title=topic['title'],
                    body=topic['body'],
                    category=topic['category']
                )
                self.logger.info(f"Created monthly discussion: {topic['title']}")
            except Exception as e:
                self.logger.error(f"Monthly discussion creation error: {e}")
    
    def create_quest_discussion_topics(self, repo):
        """
        Create quest-related discussion topics
        
        :param repo: GitHub repository
        """
        # Import Quest Generator
        sys.path.append('/home/veronicae/CascadeProjects/MLSP/modules/ai-generation/npc')
        from quest_generator import QuestGenerator
        
        # Create quest generator
        quest_gen = QuestGenerator()
        
        # Sample NPC profiles
        npc_profiles = [
            {
                'name': 'Community Quest Designer',
                'id': 'github_quest_npc_1',
                'personality_traits': ['creative', 'strategic', 'innovative']
            },
            {
                'name': 'Community Quest Tester',
                'id': 'github_quest_npc_2',
                'personality_traits': ['analytical', 'detail-oriented', 'challenging']
            }
        ]
        
        quest_topics = []
        for npc_profile in npc_profiles:
            # Generate a quest
            quest = quest_gen.generate_quest(npc_profile)
            
            # Create discussion topic
            topic = {
                'title': f"Quest Design Challenge: {quest['title']}",
                'body': f"""## Quest Design Exploration 🎲

### Quest Details
- **Title**: {quest['title']}
- **Type**: {quest['type'].capitalize()}
- **Difficulty**: {quest['difficulty'].upper()}

### Quest Description
{quest['description']}

### Objectives
{' | '.join(quest['objectives'])}

### Community Challenge
1. Analyze the quest design
2. Suggest improvements or variations
3. Share your creative quest ideas

**Rewards for Participation**: 
- Best design suggestions will be featured
- Potential implementation in future MLSP updates

Let's collaborate and push the boundaries of quest generation! 🚀""",
                'category': 'Development'
            }
            
            quest_topics.append(topic)
        
        # Create discussions
        for topic in quest_topics:
            try:
                repo.create_discussion(
                    title=topic['title'],
                    body=topic['body'],
                    category=topic['category']
                )
                self.logger.info(f"Created quest discussion: {topic['title']}")
            except Exception as e:
                self.logger.error(f"Quest discussion creation error: {e}")
    
    def run(self):
        """
        Run GitHub Discussions Manager tasks
        """
        try:
            repo_name = self.config.get('GITHUB_REPO', 'yourusername/MLSP')
            repo = self.github_client.get_repo(repo_name)
            
            # Create discussion categories
            self.create_discussion_categories(repo)
            
            # Create welcome discussion
            self.create_welcome_discussion(repo)
            
            # Create monthly discussion topics
            self.create_monthly_discussion_topics(repo)
            
            # Create quest-related discussions
            self.create_quest_discussion_topics(repo)
        
        except Exception as e:
            self.logger.error(f"GitHub Discussions management error: {e}")

def main():
    """
    Initialize and run the GitHub Discussions Manager
    """
    manager = GitHubDiscussionsManager()
    manager.run()

if __name__ == '__main__':
    main()
