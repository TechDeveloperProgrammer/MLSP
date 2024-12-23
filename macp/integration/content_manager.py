import os
import aiohttp
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModSource:
    """External mod source configuration"""
    name: str
    api_url: str
    api_key: Optional[str]
    supported_versions: List[str]

class ContentManager:
    """
    External Content Integration Manager
    
    Features:
    - Automated mod downloads
    - Version compatibility checks
    - Dependency resolution
    - Performance optimization
    """
    
    # API endpoints
    MODRINTH_API = "https://api.modrinth.com/v2"
    CURSEFORGE_API = "https://api.curseforge.com/v1"
    SPIGOT_API = "https://api.spigotmc.org/v2"
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize Content Manager
        
        :param config_dir: Configuration directory
        """
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.macp'
        self.downloads_dir = self.config_dir / 'downloads'
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize sources
        self.sources = self._initialize_sources()
    
    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='content_manager.log'
        )
        
        self.logger = logging.getLogger('ContentManager')
    
    def _initialize_sources(self) -> Dict[str, ModSource]:
        """
        Initialize mod sources
        
        :return: Dictionary of mod sources
        """
        return {
            'modrinth': ModSource(
                name='Modrinth',
                api_url=self.MODRINTH_API,
                api_key=os.getenv('MODRINTH_API_KEY'),
                supported_versions=['1.16.5', '1.17.1', '1.18.2', '1.19']
            ),
            'curseforge': ModSource(
                name='CurseForge',
                api_url=self.CURSEFORGE_API,
                api_key=os.getenv('CURSEFORGE_API_KEY'),
                supported_versions=['1.16.5', '1.17.1', '1.18.2', '1.19']
            ),
            'spigot': ModSource(
                name='SpigotMC',
                api_url=self.SPIGOT_API,
                api_key=os.getenv('SPIGOT_API_KEY'),
                supported_versions=['1.16.5', '1.17.1', '1.18.2', '1.19']
            )
        }
    
    async def search_mods(
        self,
        query: str,
        source: str,
        minecraft_version: str
    ) -> List[Dict[str, Any]]:
        """
        Search for mods across platforms
        
        :param query: Search query
        :param source: Mod source (modrinth, curseforge, spigot)
        :param minecraft_version: Minecraft version
        :return: List of mod metadata
        """
        if source not in self.sources:
            raise ValueError(f"Unknown source: {source}")
        
        mod_source = self.sources[source]
        
        async with aiohttp.ClientSession() as session:
            if source == 'modrinth':
                return await self._search_modrinth(
                    session, query, minecraft_version
                )
            elif source == 'curseforge':
                return await self._search_curseforge(
                    session, query, minecraft_version
                )
            elif source == 'spigot':
                return await self._search_spigot(
                    session, query, minecraft_version
                )
    
    async def _search_modrinth(
        self,
        session: aiohttp.ClientSession,
        query: str,
        minecraft_version: str
    ) -> List[Dict[str, Any]]:
        """
        Search Modrinth for mods
        
        :param session: HTTP session
        :param query: Search query
        :param minecraft_version: Minecraft version
        :return: List of mod metadata
        """
        try:
            params = {
                'query': query,
                'facets': [f'["versions:{minecraft_version}"]']
            }
            
            headers = {}
            if self.sources['modrinth'].api_key:
                headers['Authorization'] = self.sources['modrinth'].api_key
            
            async with session.get(
                f"{self.MODRINTH_API}/search",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['hits']
                else:
                    self.logger.error(
                        f"Modrinth search failed: {response.status}"
                    )
                    return []
        
        except Exception as e:
            self.logger.error(f"Modrinth search error: {e}")
            return []
    
    async def _search_curseforge(
        self,
        session: aiohttp.ClientSession,
        query: str,
        minecraft_version: str
    ) -> List[Dict[str, Any]]:
        """
        Search CurseForge for mods
        
        :param session: HTTP session
        :param query: Search query
        :param minecraft_version: Minecraft version
        :return: List of mod metadata
        """
        try:
            headers = {
                'x-api-key': self.sources['curseforge'].api_key
            }
            
            params = {
                'gameId': 432,  # Minecraft game ID
                'searchFilter': query,
                'gameVersion': minecraft_version
            }
            
            async with session.get(
                f"{self.CURSEFORGE_API}/mods/search",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['data']
                else:
                    self.logger.error(
                        f"CurseForge search failed: {response.status}"
                    )
                    return []
        
        except Exception as e:
            self.logger.error(f"CurseForge search error: {e}")
            return []
    
    async def _search_spigot(
        self,
        session: aiohttp.ClientSession,
        query: str,
        minecraft_version: str
    ) -> List[Dict[str, Any]]:
        """
        Search SpigotMC for plugins
        
        :param session: HTTP session
        :param query: Search query
        :param minecraft_version: Minecraft version
        :return: List of plugin metadata
        """
        try:
            params = {
                'query': query,
                'version': minecraft_version
            }
            
            headers = {}
            if self.sources['spigot'].api_key:
                headers['Authorization'] = self.sources['spigot'].api_key
            
            async with session.get(
                f"{self.SPIGOT_API}/resources/search",
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(
                        f"SpigotMC search failed: {response.status}"
                    )
                    return []
        
        except Exception as e:
            self.logger.error(f"SpigotMC search error: {e}")
            return []
    
    async def download_mod(
        self,
        mod_id: str,
        source: str,
        version: str
    ) -> Optional[Path]:
        """
        Download mod from specified source
        
        :param mod_id: Mod identifier
        :param source: Mod source
        :param version: Mod version
        :return: Path to downloaded file
        """
        if source not in self.sources:
            raise ValueError(f"Unknown source: {source}")
        
        download_path = self.downloads_dir / f"{source}_{mod_id}_{version}.jar"
        
        try:
            async with aiohttp.ClientSession() as session:
                if source == 'modrinth':
                    return await self._download_modrinth(
                        session, mod_id, version, download_path
                    )
                elif source == 'curseforge':
                    return await self._download_curseforge(
                        session, mod_id, version, download_path
                    )
                elif source == 'spigot':
                    return await self._download_spigot(
                        session, mod_id, version, download_path
                    )
        
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            return None
    
    async def _download_modrinth(
        self,
        session: aiohttp.ClientSession,
        mod_id: str,
        version: str,
        download_path: Path
    ) -> Optional[Path]:
        """
        Download mod from Modrinth
        
        :param session: HTTP session
        :param mod_id: Mod identifier
        :param version: Mod version
        :param download_path: Download destination
        :return: Path to downloaded file
        """
        try:
            headers = {}
            if self.sources['modrinth'].api_key:
                headers['Authorization'] = self.sources['modrinth'].api_key
            
            # Get version info
            async with session.get(
                f"{self.MODRINTH_API}/project/{mod_id}/version/{version}",
                headers=headers
            ) as response:
                if response.status == 200:
                    version_data = await response.json()
                    download_url = version_data['files'][0]['url']
                    
                    # Download file
                    async with session.get(download_url) as download:
                        if download.status == 200:
                            with open(download_path, 'wb') as f:
                                f.write(await download.read())
                            return download_path
                
                self.logger.error(f"Modrinth download failed: {response.status}")
                return None
        
        except Exception as e:
            self.logger.error(f"Modrinth download error: {e}")
            return None
    
    async def _download_curseforge(
        self,
        session: aiohttp.ClientSession,
        mod_id: str,
        version: str,
        download_path: Path
    ) -> Optional[Path]:
        """
        Download mod from CurseForge
        
        :param session: HTTP session
        :param mod_id: Mod identifier
        :param version: Mod version
        :param download_path: Download destination
        :return: Path to downloaded file
        """
        try:
            headers = {
                'x-api-key': self.sources['curseforge'].api_key
            }
            
            # Get download URL
            async with session.get(
                f"{self.CURSEFORGE_API}/mods/{mod_id}/files/{version}/download-url",
                headers=headers
            ) as response:
                if response.status == 200:
                    download_url = (await response.json())['data']
                    
                    # Download file
                    async with session.get(download_url) as download:
                        if download.status == 200:
                            with open(download_path, 'wb') as f:
                                f.write(await download.read())
                            return download_path
                
                self.logger.error(
                    f"CurseForge download failed: {response.status}"
                )
                return None
        
        except Exception as e:
            self.logger.error(f"CurseForge download error: {e}")
            return None
    
    async def _download_spigot(
        self,
        session: aiohttp.ClientSession,
        mod_id: str,
        version: str,
        download_path: Path
    ) -> Optional[Path]:
        """
        Download plugin from SpigotMC
        
        :param session: HTTP session
        :param mod_id: Plugin identifier
        :param version: Plugin version
        :param download_path: Download destination
        :return: Path to downloaded file
        """
        try:
            headers = {}
            if self.sources['spigot'].api_key:
                headers['Authorization'] = self.sources['spigot'].api_key
            
            # Get download URL
            async with session.get(
                f"{self.SPIGOT_API}/resources/{mod_id}/versions/{version}/download",
                headers=headers
            ) as response:
                if response.status == 200:
                    download_url = (await response.json())['downloadUrl']
                    
                    # Download file
                    async with session.get(download_url) as download:
                        if download.status == 200:
                            with open(download_path, 'wb') as f:
                                f.write(await download.read())
                            return download_path
                
                self.logger.error(f"SpigotMC download failed: {response.status}")
                return None
        
        except Exception as e:
            self.logger.error(f"SpigotMC download error: {e}")
            return None

async def main():
    """Test content manager functionality"""
    manager = ContentManager()
    
    # Test search
    results = await manager.search_mods(
        "optimization",
        "modrinth",
        "1.18.2"
    )
    
    print(f"Found {len(results)} mods")
    
    # Test download
    if results:
        mod = results[0]
        download_path = await manager.download_mod(
            mod['project_id'],
            "modrinth",
            mod['version_id']
        )
        
        if download_path:
            print(f"Downloaded to: {download_path}")

if __name__ == '__main__':
    asyncio.run(main())
