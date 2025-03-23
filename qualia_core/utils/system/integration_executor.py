"""
Repository Integration Executor
Implements the execution of repository integration decisions.
"""

import os
import shutil
from typing import Dict, Any
from .repo_integration import RepositoryIntegrator

class IntegrationExecutor:
    """
    Executes repository integration decisions.
    """
    
    def __init__(self,
                 source_repo: str,
                 target_repo: str,
                 backup: bool = True):
        self.source_repo = source_repo
        self.target_repo = target_repo
        self.backup = backup
        self.integrator = RepositoryIntegrator(source_repo, target_repo)
        
    def execute_integration(self) -> Dict[str, Any]:
        """
        Execute repository integration
        
        Returns:
            Integration results and metrics
        """
        # Create backup if requested
        if self.backup:
            self._create_backup()
            
        # Get integration decisions
        integration_results = self.integrator.integrate_repositories()
        
        # Execute decisions
        execution_results = self._execute_decisions(integration_results['decisions'])
        
        # Merge results
        final_results = {
            **integration_results,
            'execution': execution_results
        }
        
        return final_results
        
    def _create_backup(self):
        """Create backup of target repository"""
        backup_path = f"{self.target_repo}_backup"
        print(f"\nCreating backup at: {backup_path}")
        
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
            
        shutil.copytree(self.target_repo, backup_path)
        
    def _execute_decisions(self, decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integration decisions"""
        execution_results = {
            'successful_integrations': 0,
            'failed_integrations': 0,
            'skipped_files': 0,
            'errors': []
        }
        
        for rel_path, decision in decisions.items():
            if not decision['should_integrate']:
                execution_results['skipped_files'] += 1
                continue
                
            try:
                source_path = os.path.join(self.source_repo, rel_path)
                target_path = os.path.join(self.target_repo, rel_path)
                
                # Create target directory if needed
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, target_path)
                execution_results['successful_integrations'] += 1
                
            except Exception as e:
                execution_results['failed_integrations'] += 1
                execution_results['errors'].append({
                    'file': rel_path,
                    'error': str(e)
                })
                
        return execution_results
