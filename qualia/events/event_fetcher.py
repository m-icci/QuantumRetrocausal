"""
Economic Event Fetcher Service
Handles real-time economic data collection from various sources
"""
import asyncio
import aiohttp
import logging
import copy
import os  # Import os for environment variables
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from .economic_events import (
    EconomicEvent,
    EventCategory,
    EventImpact,
    EventTimelineManager
)

logger = logging.getLogger(__name__)

# Load API keys from environment variables
API_KEYS = {
    "tradingeconomics": os.getenv("TRADINGECONOMICS_API_KEY"),
    "newsapi": os.getenv("NEWSAPI_KEY"),
    "finnhub": os.getenv("FINNHUB_API_KEY"),
}

# Define new API endpoints - using example free APIs, replace with actual endpoints and consider moving to config file
EVENTS_API = {
    "economic_calendar": "https://api.tradingeconomics.com/calendar/",  # Example TradingEconomics endpoint
    "news_feed": "https://newsapi.org/v2/top-headlines?category=business&country=us",  # Example NewsAPI endpoint
    "market_events": "https://finnhub.io/api/v1/news?category=general"  # Example Finnhub endpoint
}


class EventFetcher:
    """Service for fetching economic events from multiple sources"""

    def __init__(self, timeline_manager: EventTimelineManager):
        self.timeline_manager = timeline_manager
        self.last_update = datetime.now(timezone.utc)
        self.sources = {
            'economic_calendar': self._fetch_economic_calendar,
            'news_feed': self._fetch_news_feed,
            'market_events': self._fetch_market_events
        }
        self.is_running = False
        self.fetch_loop = None
        self._quantum_dimension = 64  # Fixed dimension for quantum operations
        logger.info("Initialized EventFetcher service")


    async def _fetch_economic_calendar(self) -> List[Dict[str, Any]]:
        """Fetch economic calendar events using TradingEconomics API"""
        if API_KEYS["tradingeconomics"]:
            headers = {"Authorization": f"Bearer {API_KEYS['tradingeconomics']}"}
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(
                        EVENTS_API["economic_calendar"],
                        params={'from': self.last_update.isoformat()}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_calendar_events(data)
                        else:
                            logger.error(f"Failed to fetch calendar events: {response.status} - {await response.text()}")
                            return []
            except Exception as e:
                logger.error(f"Error fetching economic calendar: {str(e)}")
                return []
        else:
            logger.warning("No TradingEconomics API key found. Skipping economic calendar fetch.")
            return []


    async def _fetch_news_feed(self) -> List[Dict[str, Any]]:
        """Fetch financial news feed using NewsAPI"""
        if API_KEYS["newsapi"]:
            params = {'from': self.last_update.isoformat()}
            params["apiKey"] = API_KEYS["newsapi"]
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        EVENTS_API["news_feed"],
                        params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_news_events(data)
                        else:
                            logger.error(f"Failed to fetch news feed: {response.status} - {await response.text()}")
                            return []
            except Exception as e:
                logger.error(f"Error fetching news feed: {str(e)}")
                return []
        else:
            logger.warning("No NewsAPI key found. Skipping news feed fetch.")
            return []


    async def _fetch_market_events(self) -> List[Dict[str, Any]]:
        """Fetch market events using Finnhub API"""
        if API_KEYS["finnhub"]:
            params = {'from': self.last_update.isoformat()}
            params["token"] = API_KEYS["finnhub"]
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        EVENTS_API["market_events"],
                        params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_market_events(data)
                        else:
                            logger.error(f"Failed to fetch market events: {response.status} - {await response.text()}")
                            return []
            except Exception as e:
                logger.error(f"Error fetching market events: {str(e)}")
                return []
        else:
            logger.warning("No Finnhub API key found. Skipping market events fetch.")
            return []


    def _resize_pattern(self, pattern: List[float], target_dim: int = 64) -> List[float]:
        """Resize pattern to match quantum dimension"""
        if len(pattern) > target_dim:
            return pattern[:target_dim]
        elif len(pattern) < target_dim:
            return pattern + [0.0] * (target_dim - len(pattern))
        return pattern

    def _parse_calendar_events(self, data: List[Dict[str, Any]]) -> List[EconomicEvent]:
        """Parse economic calendar events into EconomicEvent objects"""
        events = []
        try:
            for event_data in data:
                # Deep copy to avoid mappingproxy issues
                safe_data = copy.deepcopy(event_data)

                # Ensure quantum patterns are correctly dimensioned
                quantum_patterns = self._resize_pattern(
                    safe_data.get('quantum_patterns', [0.0] * self._quantum_dimension)
                )

                event = EconomicEvent(
                    id=safe_data.get('id', ''),
                    title=safe_data.get('title', ''),
                    description=safe_data.get('description', ''),
                    timestamp=datetime.fromisoformat(safe_data.get('timestamp', datetime.now().isoformat())),
                    category=EventCategory(safe_data.get('category', 'economic_indicator')),
                    impact=EventImpact(safe_data.get('impact', 'unknown')),
                    region=safe_data.get('region', 'global'),
                    source='economic_calendar',
                    actual_value=safe_data.get('actual'),
                    forecast_value=safe_data.get('forecast'),
                    previous_value=safe_data.get('previous'),
                    related_assets=safe_data.get('related_assets', []),
                    additional_data={'quantum_patterns': quantum_patterns}
                )
                events.append(event)
        except Exception as e:
            logger.error(f"Error parsing calendar events: {str(e)}")
        return events

    def _parse_news_events(self, data: List[Dict[str, Any]]) -> List[EconomicEvent]:
        """Parse news feed into EconomicEvent objects"""
        events = []
        try:
            for news_item in data:
                # Deep copy to avoid mappingproxy issues
                safe_item = copy.deepcopy(news_item)

                # Ensure quantum patterns are correctly dimensioned
                quantum_patterns = self._resize_pattern(
                    safe_item.get('quantum_patterns', [0.0] * self._quantum_dimension)
                )

                event = EconomicEvent(
                    id=safe_item.get('id', ''),
                    title=safe_item.get('headline', ''),
                    description=safe_item.get('content', ''),
                    timestamp=datetime.fromisoformat(safe_item.get('published_at', datetime.now().isoformat())),
                    category=EventCategory(safe_item.get('category', 'other')),
                    impact=EventImpact(safe_item.get('impact', 'unknown')),
                    region=safe_item.get('region', 'global'),
                    source='news_feed',
                    related_assets=safe_item.get('related_assets', []),
                    additional_data={'quantum_patterns': quantum_patterns}
                )
                events.append(event)
        except Exception as e:
            logger.error(f"Error parsing news events: {str(e)}")
        return events

    def _parse_market_events(self, data: List[Dict[str, Any]]) -> List[EconomicEvent]:
        """Parse market events into EconomicEvent objects"""
        events = []
        try:
            for market_event in data:
                # Deep copy to avoid mappingproxy issues
                safe_event = copy.deepcopy(market_event)

                # Ensure quantum patterns are correctly dimensioned
                quantum_patterns = self._resize_pattern(
                    safe_event.get('quantum_patterns', [0.0] * self._quantum_dimension)
                )

                event = EconomicEvent(
                    id=safe_event.get('id', ''),
                    title=safe_event.get('title', ''),
                    description=safe_event.get('description', ''),
                    timestamp=datetime.fromisoformat(safe_event.get('timestamp', datetime.now().isoformat())),
                    category=EventCategory(safe_event.get('category', 'market_structure')),
                    impact=EventImpact(safe_event.get('impact', 'unknown')),
                    region=safe_event.get('region', 'global'),
                    source='market_events',
                    related_assets=safe_event.get('related_assets', []),
                    additional_data={
                        'quantum_patterns': quantum_patterns,
                        **safe_event.get('additional_data', {})
                    }
                )
                events.append(event)
        except Exception as e:
            logger.error(f"Error parsing market events: {str(e)}")
        return events

    async def fetch_all_events(self) -> None:
        """Fetch events from all sources"""
        try:
            tasks = [
                source_func()
                for source_func in self.sources.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in event fetching: {str(result)}")
                    continue

                if isinstance(result, list):
                    for event in result:
                        # Handle the mappingproxy issue by deep copying the event
                        safe_event = copy.deepcopy(event)
                        self.timeline_manager.add_event(safe_event)

            self.last_update = datetime.now(timezone.utc)
            logger.info("Successfully updated all events")

        except Exception as e:
            logger.error(f"Error fetching all events: {str(e)}")

    async def _fetch_loop(self, interval_seconds: int = 300):
        """Internal fetch loop"""
        while self.is_running:
            await self.fetch_all_events()
            await asyncio.sleep(interval_seconds)

    def start_fetching(self, interval_seconds: int = 300):
        """Start periodic event fetching with proper loop handling"""
        if not self.is_running:
            self.is_running = True
            try:
                # Create a new event loop for the background task
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Start the fetch loop
                self.fetch_loop = loop.run_until_complete(self._fetch_loop(interval_seconds))
                logger.info(f"Started event fetching with {interval_seconds}s interval")

            except Exception as e:
                self.is_running = False
                logger.error(f"Failed to start event fetching: {str(e)}")

    def stop_fetching(self):
        """Stop the event fetching loop"""
        self.is_running = False
        if self.fetch_loop and not self.fetch_loop.done():
            self.fetch_loop.cancel()
        logger.info("Stopped event fetching")
