"""
Enhanced training system for AI SQL generation improvement.
"""

import json
import os
import streamlit as st
from datetime import datetime
from typing import List, Dict, Optional
from supabot.config.settings import settings


class EnhancedTrainingSystem:
    """Enhanced training system for improving AI SQL generation."""
    
    def __init__(self, training_file: Optional[str] = None):
        self.training_file = training_file or settings.TRAINING_FILE
        self.training_data = self.load_training_data()
    
    def load_training_data(self) -> List[Dict]:
        """Load training examples from JSON file."""
        if os.path.exists(self.training_file):
            try:
                with open(self.training_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def save_training_data(self) -> bool:
        """Save training examples to JSON file."""
        try:
            with open(self.training_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            st.error(f"Failed to save training data: {e}")
            return False
    
    def add_training_example(self, question: str, sql: str, feedback: str = "correct", 
                           explanation: str = "") -> bool:
        """Add a new training example with optional explanation."""
        example = {
            "question": question.lower().strip(),
            "sql": sql.strip(),
            "feedback": feedback,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        self.training_data.append(example)
        return self.save_training_data()
    
    def find_similar_examples(self, question: str, limit: int = 3) -> List[Dict]:
        """Find similar training examples using enhanced similarity."""
        question = question.lower().strip()
        scored_examples = []
        
        # Business terms mapping for semantic similarity
        business_terms = {
            'sales': ['revenue', 'income', 'earnings', 'total'],
            'hour': ['time', 'hourly', 'per hour'],
            'store': ['location', 'branch', 'shop'],
            'total': ['sum', 'aggregate', 'combined', 'all'],
            'date': ['day', 'daily', 'time period']
        }
        
        for example in self.training_data:
            if example["feedback"] in ["correct", "corrected"]:
                q1_words = set(question.split())
                q2_words = set(example["question"].split())
                
                if len(q1_words | q2_words) > 0:
                    # Basic word overlap similarity
                    basic_similarity = len(q1_words & q2_words) / len(q1_words | q2_words)
                    
                    # Enhanced similarity with business terms
                    enhanced_similarity = basic_similarity
                    for term, synonyms in business_terms.items():
                        if term in q1_words:
                            for synonym in synonyms:
                                if synonym in q2_words:
                                    enhanced_similarity += 0.1
                    
                    if enhanced_similarity > 0.1:  # Minimum similarity threshold
                        scored_examples.append((enhanced_similarity, example))
        
        # Sort by similarity and return top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for _, example in scored_examples[:limit]]
    
    def get_training_context(self, question: str) -> str:
        """Generate training context for Claude based on similar examples."""
        similar_examples = self.find_similar_examples(question)
        
        if not similar_examples:
            return "No similar training examples found."
        
        context = "SIMILAR TRAINING EXAMPLES:\n"
        for i, example in enumerate(similar_examples, 1):
            context += f"\nExample {i}:\n"
            context += f"Question: {example['question']}\n"
            context += f"SQL: {example['sql']}\n"
            if example.get('explanation'):
                context += f"Explanation: {example['explanation']}\n"
        
        context += "\nUse these examples as guidance for generating the SQL query."
        return context
    
    def get_stats(self) -> Dict:
        """Get training data statistics."""
        total_examples = len(self.training_data)
        feedback_counts = {}
        
        for example in self.training_data:
            feedback = example.get('feedback', 'unknown')
            feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
        
        return {
            'total_examples': total_examples,
            'feedback_breakdown': feedback_counts,
            'latest_update': max([ex.get('timestamp', '') for ex in self.training_data]) if self.training_data else None
        }


# Global singleton instance
_training_system = None

def get_training_system() -> EnhancedTrainingSystem:
    """Get the global EnhancedTrainingSystem instance."""
    global _training_system
    if _training_system is None:
        _training_system = EnhancedTrainingSystem()
    return _training_system

