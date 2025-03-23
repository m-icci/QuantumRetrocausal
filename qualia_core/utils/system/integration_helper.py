import os
import json
import yaml
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Component:
    path: str
    type: str
    dependencies: Set[str]
    apis: Set[str]
    imports: Set[str]
    exports: Set[str]
    integrated: bool = False

class IntegrationHelper:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.components: Dict[str, Component] = {}
        self.integration_status = {}
        self.status_file = os.path.join(root_dir, "integration_tracking.yaml")
        self.load_status()

    def load_status(self):
        """Carrega o status de integração existente"""
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as f:
                self.integration_status = yaml.safe_load(f)

    def save_status(self):
        """Salva o status de integração atual"""
        with open(self.status_file, 'w') as f:
            yaml.dump(self.integration_status, f)

    def scan_codebase(self):
        """Escaneia a base de código para encontrar componentes"""
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.ts', '.tsx', '.py', '.html')):
                    path = os.path.join(root, file)
                    self.analyze_file(path)

    def analyze_file(self, path: str):
        """Analisa um arquivo para extrair informações"""
        ext = os.path.splitext(path)[1]
        relative_path = os.path.relpath(path, self.root_dir)

        if ext in ['.ts', '.tsx']:
            self._analyze_typescript(path, relative_path)
        elif ext == '.py':
            self._analyze_python(path, relative_path)
        elif ext == '.html':
            self._analyze_html(path, relative_path)

    def _analyze_typescript(self, path: str, relative_path: str):
        """Analisa arquivo TypeScript/React"""
        with open(path, 'r') as f:
            content = f.read()

        imports = set()
        apis = set()
        exports = set()

        for line in content.split('\n'):
            if line.strip().startswith('import'):
                imports.add(line.strip())
            elif line.strip().startswith('export'):
                exports.add(line.strip())
            elif 'fetch(' in line or 'axios.' in line:
                apis.add(line.strip())

        self.components[relative_path] = Component(
            path=relative_path,
            type='typescript',
            dependencies=imports,
            apis=apis,
            imports=imports,
            exports=exports,
            integrated=self._check_integration_status(relative_path)
        )

    def _analyze_python(self, path: str, relative_path: str):
        """Analisa arquivo Python"""
        with open(path, 'r') as f:
            content = f.read()

        imports = set()
        apis = set()
        exports = set()

        for line in content.split('\n'):
            if line.strip().startswith(('import', 'from')):
                imports.add(line.strip())
            elif 'requests.' in line or 'urllib' in line:
                apis.add(line.strip())
            elif line.strip().startswith(('def ', 'class ')):
                exports.add(line.strip())

        self.components[relative_path] = Component(
            path=relative_path,
            type='python',
            dependencies=imports,
            apis=apis,
            imports=imports,
            exports=exports,
            integrated=self._check_integration_status(relative_path)
        )

    def _analyze_html(self, path: str, relative_path: str):
        """Analisa arquivo HTML"""
        with open(path, 'r') as f:
            content = f.read()

        scripts = set()
        links = set()

        # Análise básica de scripts e links
        for line in content.split('\n'):
            if '<script' in line:
                scripts.add(line.strip())
            elif '<link' in line:
                links.add(line.strip())

        self.components[relative_path] = Component(
            path=relative_path,
            type='html',
            dependencies=links,
            apis=set(),
            imports=scripts,
            exports=set(),
            integrated=self._check_integration_status(relative_path)
        )

    def _check_integration_status(self, path: str) -> bool:
        """Verifica se um componente está integrado"""
        return self.integration_status.get(path, {}).get('integrated', False)

    def find_missing_integrations(self) -> List[str]:
        """Encontra componentes que precisam ser integrados"""
        missing = []
        for path, component in self.components.items():
            if not component.integrated:
                missing.append(path)
        return missing

    def suggest_integration_order(self) -> List[str]:
        """Sugere ordem de integração baseada em dependências"""
        ordered = []
        visited = set()

        def visit(path):
            if path in visited:
                return
            visited.add(path)
            component = self.components[path]
            for dep in component.dependencies:
                if dep in self.components:
                    visit(dep)
            ordered.append(path)

        for path in self.components:
            if path not in visited:
                visit(path)

        return ordered

    def generate_integration_report(self) -> Dict:
        """Gera relatório de status de integração"""
        total = len(self.components)
        integrated = len([c for c in self.components.values() if c.integrated])

        report = {
            "total_components": total,
            "integrated_components": integrated,
            "integration_progress": f"{(integrated/total)*100:.1f}%",
            "components_by_type": {},
            "missing_integrations": self.find_missing_integrations(),
            "suggested_order": self.suggest_integration_order(),
            "api_dependencies": list({api for c in self.components.values() for api in c.apis}),
            "external_dependencies": list({dep for c in self.components.values() for dep in c.dependencies})
        }

        # Contagem por tipo
        for component in self.components.values():
            if component.type not in report["components_by_type"]:
                report["components_by_type"][component.type] = 0
            report["components_by_type"][component.type] += 1

        return report

    def mark_as_integrated(self, path: str):
        """Marca um componente como integrado"""
        if path in self.components:
            self.components[path].integrated = True
            self.integration_status[path] = {"integrated": True}
            self.save_status()

    def create_integration_tasks(self) -> List[Dict]:
        """Cria lista de tarefas de integração"""
        tasks = []
        order = self.suggest_integration_order()

        for path in order:
            if not self.components[path].integrated:
                component = self.components[path]
                tasks.append({
                    "path": path,
                    "type": component.type,
                    "dependencies": list(component.dependencies),
                    "apis": list(component.apis),
                    "priority": "high" if not component.dependencies else "medium"
                })

        return tasks

if __name__ == "__main__":
    helper = IntegrationHelper("/Users/infrastructure/Documents/GitHub/ICCI")
    helper.scan_codebase()
    
    # Gerar relatório
    report = helper.generate_integration_report()
    print("\n=== Relatório de Integração ===")
    print(f"Total de Componentes: {report['total_components']}")
    print(f"Componentes Integrados: {report['integrated_components']}")
    print(f"Progresso: {report['integration_progress']}")
    
    print("\nComponentes por Tipo:")
    for type_, count in report["components_by_type"].items():
        print(f"- {type_}: {count}")
    
    print("\nPróximos Componentes para Integrar:")
    for path in report["missing_integrations"][:5]:
        print(f"- {path}")
    
    # Criar tarefas
    tasks = helper.create_integration_tasks()
    
    # Salvar relatório e tarefas
    with open('integration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    with open('integration_tasks.json', 'w') as f:
        json.dump(tasks, f, indent=2)