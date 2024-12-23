import os
import asyncio
import logging
import discord
from discord.ext import commands, tasks
import aiohttp
import json
from datetime import datetime, timedelta
import sys

class MLSPCommunityBot(commands.Bot):
    """
    Minecraft Local Server Platform Community Discord Bot
    
    Features:
    - Project update notifications
    - Community engagement tools
    - Automated moderation
    - GitHub integration
    - Support ticket management
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the MLSP Community Bot
        
        :param args: Positional arguments for Bot initialization
        :param kwargs: Keyword arguments for Bot initialization
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        super().__init__(
            command_prefix='!mlsp ', 
            intents=intents,
            *args, 
            **kwargs
        )
        
        # Configuration
        self.config = self.load_config()
        
        # Logging
        self.setup_logging()
        
        # GitHub integration
        self.github_token = self.config.get('GITHUB_TOKEN')
        self.github_repo = self.config.get('GITHUB_REPO', 'yourusername/MLSP')
        
        # Background tasks
        self.github_update_task = None
        self.project_metrics_task = None
    
    def load_config(self):
        """
        Load bot configuration
        
        :return: Configuration dictionary
        """
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'config', 
            'bot_config.json'
        )
        
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
            filename='mlsp_discord_bot.log'
        )
        
        self.logger = logging.getLogger('MLSPCommunityBot')
    
    async def on_ready(self):
        """
        Bot startup routine
        """
        self.logger.info(f'Logged in as {self.user.name}')
        
        # Start background tasks
        self.github_update_task = self.start_github_updates()
        self.project_metrics_task = self.start_project_metrics()
    
    @tasks.loop(hours=24)
    async def start_github_updates(self):
        """
        Fetch and announce GitHub updates
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch latest releases
                async with session.get(
                    f'https://api.github.com/repos/{self.github_repo}/releases/latest',
                    headers={'Authorization': f'token {self.github_token}'}
                ) as response:
                    if response.status == 200:
                        release = await response.json()
                        await self.announce_release(release)
                
                # Fetch recent issues
                async with session.get(
                    f'https://api.github.com/repos/{self.github_repo}/issues?state=open',
                    headers={'Authorization': f'token {self.github_token}'}
                ) as response:
                    if response.status == 200:
                        issues = await response.json()
                        await self.announce_issues(issues)
        
        except Exception as e:
            self.logger.error(f"GitHub update error: {e}")
    
    async def announce_release(self, release):
        """
        Announce new GitHub release
        
        :param release: Release metadata
        """
        channel = self.get_channel(self.config.get('ANNOUNCEMENT_CHANNEL_ID'))
        if channel:
            embed = discord.Embed(
                title=f"🚀 New Release: {release['tag_name']}",
                description=release['body'],
                url=release['html_url'],
                color=discord.Color.green()
            )
            await channel.send(embed=embed)
    
    async def announce_issues(self, issues):
        """
        Announce recent GitHub issues
        
        :param issues: List of open issues
        """
        channel = self.get_channel(self.config.get('DISCUSSION_CHANNEL_ID'))
        if channel:
            for issue in issues[:5]:  # Top 5 recent issues
                embed = discord.Embed(
                    title=f"🐛 New Issue: {issue['title']}",
                    description=issue['body'][:200] + '...',
                    url=issue['html_url'],
                    color=discord.Color.red()
                )
                await channel.send(embed=embed)
    
    @tasks.loop(hours=12)
    async def start_project_metrics(self):
        """
        Collect and announce project metrics
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch repository metrics
                async with session.get(
                    f'https://api.github.com/repos/{self.github_repo}',
                    headers={'Authorization': f'token {self.github_token}'}
                ) as response:
                    if response.status == 200:
                        repo_data = await response.json()
                        await self.announce_metrics(repo_data)
        
        except Exception as e:
            self.logger.error(f"Project metrics error: {e}")
    
    async def announce_metrics(self, repo_data):
        """
        Announce project metrics
        
        :param repo_data: Repository metadata
        """
        channel = self.get_channel(self.config.get('METRICS_CHANNEL_ID'))
        if channel:
            embed = discord.Embed(
                title="📊 MLSP Project Metrics",
                color=discord.Color.blue()
            )
            embed.add_field(
                name="Stars ⭐", 
                value=repo_data.get('stargazers_count', 'N/A'), 
                inline=True
            )
            embed.add_field(
                name="Forks 🍴", 
                value=repo_data.get('forks_count', 'N/A'), 
                inline=True
            )
            embed.add_field(
                name="Open Issues 🐞", 
                value=repo_data.get('open_issues_count', 'N/A'), 
                inline=True
            )
            await channel.send(embed=embed)
    
    @commands.command(name='help')
    async def custom_help(self, ctx):
        """
        Custom help command for MLSP Discord Bot
        
        :param ctx: Command context
        """
        help_embed = discord.Embed(
            title="MLSP Community Bot Commands",
            description="Here are the available commands:",
            color=discord.Color.purple()
        )
        help_embed.add_field(
            name="!mlsp help", 
            value="Display this help message", 
            inline=False
        )
        help_embed.add_field(
            name="!mlsp status", 
            value="Check current project status", 
            inline=False
        )
        help_embed.add_field(
            name="!mlsp contribute", 
            value="Get information about contributing", 
            inline=False
        )
        help_embed.add_field(
            name="!mlsp quest", 
            value="Generate a sample quest for community interaction", 
            inline=False
        )
        await ctx.send(embed=help_embed)
    
    @commands.command(name='status')
    async def project_status(self, ctx):
        """
        Display current project status
        
        :param ctx: Command context
        """
        status_embed = discord.Embed(
            title="MLSP Project Status",
            description="Current development status and upcoming features",
            color=discord.Color.orange()
        )
        status_embed.add_field(
            name="Version", 
            value="0.1.0-alpha", 
            inline=True
        )
        status_embed.add_field(
            name="Active Development", 
            value="AI Generation, Performance Optimization", 
            inline=True
        )
        status_embed.add_field(
            name="Next Milestone", 
            value="NPC Interaction Improvements", 
            inline=False
        )
        await ctx.send(embed=status_embed)
    
    @commands.command(name='contribute')
    async def contribution_info(self, ctx):
        """
        Provide contribution guidelines
        
        :param ctx: Command context
        """
        contrib_embed = discord.Embed(
            title="Contribute to MLSP",
            description="Ways you can help improve the project",
            color=discord.Color.green()
        )
        contrib_embed.add_field(
            name="Code Contributions", 
            value="Check our GitHub repository for open issues", 
            inline=False
        )
        contrib_embed.add_field(
            name="Documentation", 
            value="Help improve our docs and tutorials", 
            inline=False
        )
        contrib_embed.add_field(
            name="Testing", 
            value="Report bugs and suggest improvements", 
            inline=False
        )
        await ctx.send(embed=contrib_embed)
    
    @commands.command(name='quest')
    async def generate_quest(self, ctx, difficulty: str = 'medium'):
        """
        Generate a sample quest for community interaction
        
        :param ctx: Command context
        :param difficulty: Quest difficulty level
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
                'id': 'discord_community_npc',
                'personality_traits': ['helpful', 'adventurous', 'curious']
            }
            
            # Generate quest
            quest = quest_gen.generate_quest(npc_profile)
            
            # Create Discord embed for quest
            quest_embed = discord.Embed(
                title=f"🎲 Quest: {quest['title']}",
                description=quest['description'],
                color=discord.Color.gold()
            )
            
            quest_embed.add_field(
                name="Difficulty", 
                value=quest['difficulty'].capitalize(), 
                inline=True
            )
            quest_embed.add_field(
                name="Type", 
                value=quest['type'].capitalize(), 
                inline=True
            )
            
            # Add objectives
            objectives_text = "\n".join(
                [f"- {obj}" for obj in quest['objectives']]
            )
            quest_embed.add_field(
                name="Objectives", 
                value=objectives_text, 
                inline=False
            )
            
            # Add rewards
            rewards_text = "\n".join([
                f"- {reward['type']}: {reward['value']}" 
                for reward in quest['rewards']
            ])
            quest_embed.add_field(
                name="Rewards", 
                value=rewards_text, 
                inline=False
            )
            
            await ctx.send(embed=quest_embed)
        
        except Exception as e:
            await ctx.send(f"Quest generation error: {e}")

def main():
    """
    Initialize and run the MLSP Community Bot
    """
    bot = MLSPCommunityBot()
    
    # Load bot token from configuration
    config = bot.load_config()
    bot_token = config.get('DISCORD_BOT_TOKEN')
    
    if not bot_token:
        logging.error("No Discord bot token found in configuration")
        return
    
    bot.run(bot_token)

if __name__ == '__main__':
    main()
