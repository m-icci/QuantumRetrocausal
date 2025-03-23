from typing import Dict, Any, List
from dataclasses import dataclass
import wolframclient.language as wl
from wolframclient.evaluation import WolframLanguageSession

@dataclass
class WolframExpression:
    code: str
    result: Any
    metadata: Dict[str, Any]

@dataclass
class WolframPattern:
    expression: str
    rules: List[str]
    transformations: List[str]
    metadata: Dict[str, Any]

class WolframBridge:
    def __init__(self):
        self.session = WolframLanguageSession()
        
    def evaluate_expression(self, expression: str) -> WolframExpression:
        """Evaluates a Wolfram expression"""
        try:
            # Convert expression to Wolfram code
            code = self._prepare_expression(expression)
            
            # Evaluate expression
            result = self.session.evaluate(code)
            
            return WolframExpression(
                code=code,
                result=result,
                metadata={}
            )
            
        except Exception as e:
            raise Exception(f"Error evaluating expression: {str(e)}")
        
    def find_patterns(self, data: Any) -> List[WolframPattern]:
        """Finds patterns in data using Wolfram's pattern matching"""
        patterns = []
        
        try:
            # Convert data to Wolfram format
            wl_data = self._convert_to_wolfram(data)
            
            # Find patterns
            pattern_expr = wl.FindPattern(wl_data)
            result = self.session.evaluate(pattern_expr)
            
            # Convert results to patterns
            patterns = self._convert_to_patterns(result)
            
        except Exception as e:
            raise Exception(f"Error finding patterns: {str(e)}")
            
        return patterns
        
    def apply_rules(self, expression: str, rules: List[str]) -> WolframExpression:
        """Applies transformation rules to expression"""
        try:
            # Convert expression and rules to Wolfram format
            wl_expr = self._prepare_expression(expression)
            wl_rules = [self._prepare_expression(rule) for rule in rules]
            
            # Apply rules
            result = self.session.evaluate(
                wl.ReplaceAll(wl_expr, wl_rules)
            )
            
            return WolframExpression(
                code=str(wl_expr),
                result=result,
                metadata={'rules': rules}
            )
            
        except Exception as e:
            raise Exception(f"Error applying rules: {str(e)}")
        
    def _prepare_expression(self, expression: str) -> Any:
        """Prepares expression for Wolfram evaluation"""
        try:
            # Parse expression
            parsed = wl.ToExpression(expression)
            
            # Validate expression
            if not self._is_valid_expression(parsed):
                raise ValueError("Invalid expression")
                
            return parsed
            
        except Exception as e:
            raise Exception(f"Error preparing expression: {str(e)}")
        
    def _convert_to_wolfram(self, data: Any) -> Any:
        """Converts Python data to Wolfram format"""
        if isinstance(data, (int, float, str)):
            return data
            
        elif isinstance(data, list):
            return wl.List(*[self._convert_to_wolfram(x) for x in data])
            
        elif isinstance(data, dict):
            return wl.Association(*[
                wl.Rule(k, self._convert_to_wolfram(v))
                for k, v in data.items()
            ])
            
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
    def _convert_to_patterns(self, result: Any) -> List[WolframPattern]:
        """Converts Wolfram pattern results to WolframPattern objects"""
        patterns = []
        
        try:
            # Extract pattern information
            for pattern_data in result:
                pattern = WolframPattern(
                    expression=str(pattern_data[0]),
                    rules=[str(r) for r in pattern_data[1]],
                    transformations=[str(t) for t in pattern_data[2]],
                    metadata={}
                )
                patterns.append(pattern)
                
        except Exception as e:
            raise Exception(f"Error converting patterns: {str(e)}")
            
        return patterns
        
    def _is_valid_expression(self, expression: Any) -> bool:
        """Validates Wolfram expression"""
        try:
            # Check expression validity
            result = self.session.evaluate(
                wl.SyntaxQ(expression)
            )
            return bool(result)
            
        except:
            return False
        
    def close(self):
        """Closes Wolfram session"""
        try:
            self.session.terminate()
        except:
            pass
