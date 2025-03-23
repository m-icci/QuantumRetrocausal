import os
import shutil
from pathlib import Path
import json
from typing import Dict, List, Set
import hashlib
from datetime import datetime

class IntegrationManager:
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.categories = {
            'source_code': {
                'patterns': ['*.py', '*.ts', '*.js', '*.tsx', '*.jsx'],
                'target_dirs': {
                    'frontend/app': 'frontend/app',
                    'backend': 'backend',
                    'api': 'api',
                    'ml': 'ml',
                    'core': 'core',
                    'mobile/src': 'mobile/src',
                    'trading': 'trading'
                }
            },
            'documentation': {
                'patterns': ['*.md', '*.txt', '*.pdf', '*.ipynb', '*.rst'],
                'target_dirs': {
                    'docs': 'docs',
                    'notebooks': 'notebooks'
                }
            },
            'configuration': {
                'patterns': ['*.yml', '*.yaml', '*.json', '*.ini', '*.cfg', '*.conf', 'requirements.txt', 'setup.py'],
                'target_dirs': {
                    'config': 'config',
                    'deployment': 'deployment'
                }
            },
            'tests': {
                'patterns': ['test_*.py', '*_test.py', '*.spec.ts'],
                'target_dirs': {
                    'tests': 'tests'
                }
            }
        }
        self.stats = {cat: {'total': 0, 'copied': 0} for cat in self.categories}
        
    def categorize_file(self, file_path: Path) -> str:
        """Determine the category of a file based on its extension and path."""
        for category, info in self.categories.items():
            # Check if file matches any pattern in the category
            if any(file_path.match(pattern) for pattern in info['patterns']):
                # Check if file is in any of the category's target directories
                for source_dir in info['target_dirs']:
                    if source_dir in str(file_path):
                        return category
        return 'other'

    def get_target_path(self, source_path: Path, category: str) -> Path:
        """Determine the target path for a file based on its category and current location."""
        rel_path = source_path.relative_to(self.source_dir)
        
        # Try to map to specific target directory
        for source_dir, target_dir in self.categories[category]['target_dirs'].items():
            if source_dir in str(rel_path):
                new_path = str(rel_path).replace(source_dir, target_dir)
                return self.target_dir / new_path
        
        # Default fallback - maintain relative path
        return self.target_dir / rel_path

    def copy_file(self, source: Path, target: Path) -> bool:
        """Copy a file and verify the copy was successful."""
        try:
            # Create target directory if it doesn't exist
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source, target)
            
            # Verify copy was successful
            if source.stat().st_size == target.stat().st_size:
                return True
            return False
        except Exception as e:
            print(f"Error copying {source} to {target}: {str(e)}")
            return False

    def integrate_category(self, category: str) -> Dict:
        """Integrate all files of a specific category."""
        results = {
            'successful': [],
            'failed': [],
            'total': 0
        }
        
        print(f"\nIntegrating {category} files...")
        
        # Find all files matching category patterns
        for pattern in self.categories[category]['patterns']:
            for source_path in self.source_dir.rglob(pattern):
                if source_path.is_file():
                    results['total'] += 1
                    
                    # Get target path
                    target_path = self.get_target_path(source_path, category)
                    
                    # Copy file
                    if self.copy_file(source_path, target_path):
                        results['successful'].append(str(source_path))
                        print(f"✓ Copied: {source_path.relative_to(self.source_dir)}")
                    else:
                        results['failed'].append(str(source_path))
                        print(f"✗ Failed: {source_path.relative_to(self.source_dir)}")
        
        return results

    def run_integration(self):
        """Run the complete integration process."""
        start_time = datetime.now()
        
        print("Starting Integration Process...")
        
        results = {}
        for category in self.categories:
            results[category] = self.integrate_category(category)
        
        # Generate report
        report = {
            'timestamp': start_time.isoformat(),
            'duration': str(datetime.now() - start_time),
            'results': results,
            'summary': {
                'total_files': sum(r['total'] for r in results.values()),
                'successful': sum(len(r['successful']) for r in results.values()),
                'failed': sum(len(r['failed']) for r in results.values())
            }
        }
        
        # Save report
        report_path = self.target_dir / 'integration_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nIntegration Summary:")
        print(f"Total files processed: {report['summary']['total_files']}")
        print(f"Successfully copied: {report['summary']['successful']}")
        print(f"Failed to copy: {report['summary']['failed']}")
        print(f"\nDetailed report saved to: {report_path}")

def main():
    source_dir = "/Users/infrastructure/Downloads/YAA"
    target_dir = "/Users/infrastructure/Desktop/YAA/yaa_quantum"
    
    manager = IntegrationManager(source_dir, target_dir)
    manager.run_integration()

if __name__ == "__main__":
    main()
