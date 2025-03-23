"""
Blockchain integration module for CGR analysis.
Handles smart contract interaction and data collection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from .cgr_metrics import CGRMetricsAnalyzer
from .advanced_cgr import AdvancedCGR
from .market_cosmos import MarketCosmicAnalyzer
from .field import QuantumField
from .cache_manager import CacheManager
from .chain_adapter import ChainAdapterFactory

@dataclass
class BlockchainConfig:
    """Configuration for blockchain integration."""
    chain_id: int
    contract_address: str
    abi_path: str
    cache_dir: str = 'cache'
    start_block: int = 0
    batch_size: int = 1000
    n_symbols: int = 6
    scaling_factor: float = 0.5
    quantum_coupling: float = 0.1

class BlockchainCGRIntegration:
    """
    Integrates blockchain data with CGR analysis.
    """
    
    def __init__(self, config: BlockchainConfig):
        """Initialize blockchain integration."""
        self.config = config
        
        # Initialize blockchain adapter
        self.chain = ChainAdapterFactory.create_adapter(config.chain_id)
        if not self.chain.connect():
            raise ConnectionError(f"Failed to connect to chain {config.chain_id}")
        
        # Load contract ABI
        with open(config.abi_path, 'r') as f:
            contract_abi = f.read()
        
        self.contract = self.chain.get_contract(
            address=config.contract_address,
            abi=contract_abi
        )
        
        # Initialize components
        self.cgr = AdvancedCGR()
        self.metrics = CGRMetricsAnalyzer()
        self.quantum_field = QuantumField()
        self.cosmic = MarketCosmicAnalyzer()
        self.cache = CacheManager(config.cache_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def collect_events(self, 
                      event_name: str,
                      from_block: Optional[int] = None,
                      to_block: Optional[int] = None,
                      use_cache: bool = True) -> List[Dict]:
        """
        Collect blockchain events for analysis.
        
        Args:
            event_name: Name of the event to collect
            from_block: Starting block number
            to_block: Ending block number
            use_cache: Whether to use cached results
            
        Returns:
            List of event logs
        """
        if from_block is None:
            from_block = self.config.start_block
        if to_block is None:
            to_block = self.chain.get_latest_block()
            
        self.logger.info(f"Collecting {event_name} events from block {from_block} to {to_block}")
        
        # Check cache first
        if use_cache:
            cached_events = self.cache.get_events(
                chain_id=self.config.chain_id,
                contract_address=self.config.contract_address,
                event_name=event_name
            )
            if cached_events is not None:
                self.logger.info(f"Using cached events for {event_name}")
                return cached_events
        
        # Collect events in batches
        all_events = []
        current_block = from_block
        
        while current_block < to_block:
            batch_end = min(current_block + self.config.batch_size, to_block)
            
            events = self.chain.get_events(
                contract=self.contract,
                event_name=event_name,
                from_block=current_block,
                to_block=batch_end
            )
            
            all_events.extend(events)
            current_block = batch_end + 1
            
        # Cache results
        if use_cache:
            self.cache.store_events(
                chain_id=self.config.chain_id,
                contract_address=self.config.contract_address,
                events=all_events,
                event_name=event_name
            )
        
        return all_events
    
    def normalize_events(self, events: List[Dict]) -> np.ndarray:
        """
        Normalize blockchain events for CGR analysis.
        
        Args:
            events: List of event logs
            
        Returns:
            Normalized event data
        """
        # Extract numerical values
        values = []
        for event in events:
            # Add transaction value
            values.append(float(event['args'].get('value', 0)))
            # Add gas price
            values.append(float(self.chain.get_transaction(
                event['transactionHash']
            )['gasPrice']))
            
        # Convert to numpy array
        values = np.array(values)
        
        # Normalize to [0, 1]
        if len(values) > 0:
            values = (values - np.min(values)) / (np.max(values) - np.min(values))
            
        return values
    
    def discretize_events(self, 
                         normalized_data: np.ndarray,
                         n_bins: int = 6) -> np.ndarray:
        """
        Discretize normalized events into symbols.
        
        Args:
            normalized_data: Normalized event data
            n_bins: Number of bins for discretization
            
        Returns:
            Discretized symbols
        """
        bins = np.linspace(0, 1, n_bins + 1)
        return np.digitize(normalized_data, bins) - 1
    
    def analyze_contract_state(self,
                             event_name: str,
                             window_size: int = 1000,
                             use_cache: bool = True) -> Dict:
        """
        Analyze contract state using CGR.
        
        Args:
            event_name: Name of the event to analyze
            window_size: Number of recent events to analyze
            use_cache: Whether to use cached results
            
        Returns:
            Analysis results
        """
        # Check cache first
        if use_cache:
            cached_results = self.cache.get_cgr_results(
                chain_id=self.config.chain_id,
                contract_address=self.config.contract_address,
                event_name=event_name
            )
            if cached_results is not None:
                self.logger.info("Using cached CGR results")
                return cached_results
        
        # Collect recent events
        latest_block = self.chain.get_latest_block()
        from_block = latest_block - window_size
        
        events = self.collect_events(
            event_name=event_name,
            from_block=from_block,
            to_block=latest_block,
            use_cache=use_cache
        )
        
        # Process events
        normalized_data = self.normalize_events(events)
        discretized_data = self.discretize_events(
            normalized_data,
            n_bins=self.config.n_symbols
        )
        
        # Generate CGR
        cgr_points = self.cgr.generate_cgr(
            discretized_data,
            scaling_factor=self.config.scaling_factor
        )
        cgr_matrix = self.cgr.calculate_density_matrix(cgr_points)
        
        # Calculate metrics
        metrics = self.metrics.calculate_all_metrics(
            market_data=normalized_data,
            cgr_matrix=cgr_matrix,
            cgr_points=cgr_points,
            patterns=self.cgr.detect_patterns(cgr_points)
        )
        
        # Get quantum field influence
        field_strength = self.quantum_field.get_field_strength()
        
        results = {
            'metrics': metrics,
            'field_strength': field_strength,
            'latest_block': latest_block,
            'events_analyzed': len(events)
        }
        
        # Cache results
        if use_cache:
            self.cache.store_cgr_results(
                cgr_points=cgr_points,
                cgr_matrix=cgr_matrix,
                metrics=metrics.__dict__,
                patterns=self.cgr.detect_patterns(cgr_points),
                cache_key=self._get_cache_key(event_name)
            )
        
        return results
    
    def _get_cache_key(self, event_name: str) -> str:
        """Generate cache key for event."""
        return self.cache._get_cache_key(
            'events',
            chain_id=self.config.chain_id,
            contract_address=self.config.contract_address,
            event_name=event_name
        )
    
    def adjust_contract_parameters(self,
                                 analysis_results: Dict,
                                 gas_limit: Optional[int] = None,
                                 fees: Optional[int] = None) -> Dict:
        """
        Adjust contract parameters based on CGR analysis.
        
        Args:
            analysis_results: Results from analyze_contract_state
            gas_limit: Optional gas limit override
            fees: Optional fees override
            
        Returns:
            Transaction details
        """
        # Calculate optimal parameters based on metrics
        metrics = analysis_results['metrics']
        field_strength = analysis_results['field_strength']
        
        if gas_limit is None:
            # Adjust gas limit based on pattern density and market efficiency
            base_gas = 100000
            pattern_factor = metrics.pattern_density
            efficiency_factor = metrics.market_efficiency
            gas_limit = int(base_gas * (1 + pattern_factor) * (1 + efficiency_factor))
            
        if fees is None:
            # Adjust fees based on quantum metrics
            base_fee = self.chain.get_gas_price()
            quantum_factor = metrics.quantum_coherence
            entropy_factor = metrics.quantum_entropy
            fees = int(base_fee * (1 + quantum_factor) * (1 + entropy_factor))
        
        # Build transaction
        txn = self.contract.functions.setParams(
            gasLimit=gas_limit,
            fees=fees
        ).buildTransaction({
            'from': self.chain.get_account(),
            'gas': 2000000,
            'gasPrice': self.chain.get_gas_price(),
            'nonce': self.chain.get_transaction_count(
                self.chain.get_account()
            )
        })
        
        # Sign and send transaction
        signed_txn = self.chain.sign_transaction(
            txn,
            private_key='<PRIVATE_KEY>'
        )
        tx_hash = self.chain.send_raw_transaction(signed_txn.rawTransaction)
        
        return {
            'transaction_hash': tx_hash.hex(),
            'gas_limit': gas_limit,
            'fees': fees,
            'quantum_influence': field_strength
        }
