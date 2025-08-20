import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Dict, Optional
import random

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrystalRAG:
    """
    Enhanced Crystal RAG Engine for celira's Mystical Crystal Healing Chatbot
    
    This class provides intelligent crystal recommendations based on semantic similarity
    using sentence transformers and FAISS for efficient vector search.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the Crystal RAG engine with enhanced error handling and logging.
        
        Args:
            csv_path (str): Path to the crystal dataset CSV file
        """
        try:
            logger.info("🔮 Initializing celira's Crystal RAG Engine...")
            
            # Load and preprocess crystal dataset
            self.df = pd.read_csv(csv_path).fillna("")
            logger.info(f"✨ Loaded {len(self.df)} crystals into the mystical database")
            
            # Create enriched text representations for each crystal
            self.text_chunks = self.df.apply(self._row_to_enhanced_text, axis=1).tolist()
            
            # Initialize sentence transformer model for semantic understanding
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("🧚‍♀️ Sentence transformer model loaded successfully")
            
            # Generate embeddings for all crystal descriptions
            logger.info("🌟 Generating crystal energy embeddings...")
            self.embeddings = self.model.encode(
                self.text_chunks, 
                convert_to_numpy=True, 
                show_progress_bar=True
            )
            
            # Build FAISS index for fast similarity search
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            
            logger.info("💎 Crystal RAG engine initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Crystal RAG: {str(e)}")
            raise

    def _row_to_enhanced_text(self, row: pd.Series) -> str:
        """
        Convert a crystal data row to enhanced, searchable text representation.
        
        Args:
            row (pd.Series): A row from the crystal dataset
            
        Returns:
            str: Enhanced text representation of the crystal
        """
        # Create a comprehensive text description with mystical language
        text_parts = []
        
        # Crystal name and basic info
        if row.get('Name'):
            text_parts.append(f"Crystal Name: {row['Name']}")
        
        # Chakra associations with mystical context
        if row.get('Chakra_Association'):
            text_parts.append(f"Chakra Energy: {row['Chakra_Association']} chakra alignment")
        
        # Healing properties with emotional context
        if row.get('Helps_With'):
            text_parts.append(f"Healing Support: {row['Helps_With']}")
        
        # Detailed healing properties
        if row.get('Healing_Properties'):
            text_parts.append(f"Mystical Properties: {row['Healing_Properties']}")
        
        # Color energy
        if row.get('Color'):
            text_parts.append(f"Color Energy: {row['Color']} vibrations")
        
        # Usage guidance
        if row.get('Usage_Tips'):
            text_parts.append(f"Sacred Usage: {row['Usage_Tips']}")
        
        # Affirmations for manifestation
        if row.get('Affirmation'):
            text_parts.append(f"Crystal Affirmation: {row['Affirmation']}")
        
        # Additional mystical keywords for better matching
        mystical_keywords = self._generate_mystical_keywords(row)
        if mystical_keywords:
            text_parts.append(f"Energy Keywords: {mystical_keywords}")
        
        return "\n".join(text_parts)

    def _generate_mystical_keywords(self, row: pd.Series) -> str:
        """
        Generate additional mystical keywords based on crystal properties.
        
        Args:
            row (pd.Series): Crystal data row
            
        Returns:
            str: Additional mystical keywords
        """
        keywords = []
        
        # Map common healing needs to mystical terms
        helps_with = str(row.get('Helps_With', '')).lower()
        
        if 'anxiety' in helps_with or 'stress' in helps_with:
            keywords.extend(['calming energy', 'peaceful vibrations', 'serenity'])
        
        if 'love' in helps_with or 'heart' in helps_with:
            keywords.extend(['heart opening', 'unconditional love', 'emotional healing'])
        
        if 'protection' in helps_with:
            keywords.extend(['spiritual shield', 'negative energy clearing', 'aura protection'])
        
        if 'clarity' in helps_with or 'focus' in helps_with:
            keywords.extend(['mental clarity', 'spiritual insight', 'divine wisdom'])
        
        if 'abundance' in helps_with or 'prosperity' in helps_with:
            keywords.extend(['manifestation', 'wealth energy', 'success vibrations'])
        
        # Add chakra-specific keywords
        chakra = str(row.get('Chakra_Association', '')).lower()
        chakra_keywords = {
            'root': ['grounding', 'stability', 'earth energy', 'survival'],
            'sacral': ['creativity', 'passion', 'sexuality', 'emotional flow'],
            'solar': ['confidence', 'personal power', 'willpower', 'self-esteem'],
            'heart': ['love', 'compassion', 'forgiveness', 'emotional healing'],
            'throat': ['communication', 'truth', 'expression', 'speaking'],
            'third eye': ['intuition', 'psychic abilities', 'inner wisdom', 'spiritual sight'],
            'crown': ['spiritual connection', 'divine consciousness', 'enlightenment', 'cosmic awareness']
        }
        
        for chakra_name, chakra_words in chakra_keywords.items():
            if chakra_name in chakra:
                keywords.extend(chakra_words)
        
        return ', '.join(keywords)

    def query(self, user_input: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Enhanced query method that returns diverse, structured crystal recommendations.
        Prevents repetition of frequently occurring crystals like Serpentine.
        """
        try:
            logger.info(f"🔍 Searching for crystals matching: '{user_input}'")

            # Enhance user input with mystical context
            enhanced_input = self._enhance_user_query(user_input)

            # Generate embedding for user query
            input_vec = self.model.encode([enhanced_input], convert_to_numpy=True)

            # Search for similar crystals
            scores, indices = self.index.search(input_vec, 10)  # Fetch more than top_k to allow filtering

            seen_names = set()
            recommendations = []

            for i, idx in enumerate(indices[0]):
                if idx >= len(self.df):
                    continue

                crystal_data = self.df.iloc[idx]
                name = crystal_data.get('Name', 'Unknown Crystal')

                # ✋ Skip duplicates and over-suggested crystals
                if name.lower() in seen_names or name.lower() == "serpentine":
                    continue
                seen_names.add(name.lower())

                # ✨ Apply similarity score threshold (optional)
                if float(scores[0][i]) > 1.0:  # Lower score = more similar
                    continue

                recommendation = {
                    'name': name,
                    'chakra': crystal_data.get('Chakra_Association', 'Universal'),
                    'helps_with': crystal_data.get('Helps_With', 'General healing'),
                    'properties': crystal_data.get('Healing_Properties', 'Mystical energy'),
                    'color': crystal_data.get('Color', 'Rainbow'),
                    'usage': crystal_data.get('Usage_Tips', 'Hold with intention'),
                    'affirmation': crystal_data.get('Affirmation', 'I am aligned with healing energy'),
                    'similarity_score': float(scores[0][i]),
                    'full_text': self.text_chunks[idx]
                }

                recommendations.append(recommendation)
                if len(recommendations) >= top_k:
                    break

            logger.info(f"✨ Returning {len(recommendations)} diverse crystal recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"❌ Error during crystal search: {str(e)}")
            return []

    def _enhance_user_query(self, user_input: str) -> str:
        """
        Enhance user query with mystical context for better matching.
        
        Args:
            user_input (str): Original user input
            
        Returns:
            str: Enhanced query with mystical context
        """
        # Add mystical context to improve matching
        mystical_context = [
            "crystal healing",
            "spiritual guidance",
            "energy alignment",
            "chakra balancing",
            "mystical properties"
        ]
        
        enhanced_query = f"{user_input} {' '.join(mystical_context)}"
        return enhanced_query

    def get_crystal_by_name(self, crystal_name: str) -> Optional[Dict[str, str]]:
        """
        Get detailed information about a specific crystal by name.
        
        Args:
            crystal_name (str): Name of the crystal to look up
            
        Returns:
            Optional[Dict[str, str]]: Crystal information or None if not found
        """
        try:
            # Case-insensitive search for crystal name
            crystal_row = self.df[self.df['Name'].str.lower() == crystal_name.lower()]
            
            if not crystal_row.empty:
                crystal_data = crystal_row.iloc[0]
                return {
                    'name': crystal_data.get('Name', 'Unknown Crystal'),
                    'chakra': crystal_data.get('Chakra_Association', 'Universal'),
                    'helps_with': crystal_data.get('Helps_With', 'General healing'),
                    'properties': crystal_data.get('Healing_Properties', 'Mystical energy'),
                    'color': crystal_data.get('Color', 'Rainbow'),
                    'usage': crystal_data.get('Usage_Tips', 'Hold with intention'),
                    'affirmation': crystal_data.get('Affirmation', 'I am aligned with healing energy')
                }
            else:
                logger.warning(f"🔍 Crystal '{crystal_name}' not found in database")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error looking up crystal '{crystal_name}': {str(e)}")
            return None

    def get_chakra_crystals(self, chakra_name: str) -> List[Dict[str, str]]:
        """
        Get all crystals associated with a specific chakra.
        
        Args:
            chakra_name (str): Name of the chakra
            
        Returns:
            List[Dict[str, str]]: List of crystals for the specified chakra
        """
        try:
            # Filter crystals by chakra association
            chakra_crystals = self.df[
                self.df['Chakra_Association'].str.lower().str.contains(
                    chakra_name.lower(), na=False
                )
            ]
            
            recommendations = []
            for _, crystal_data in chakra_crystals.iterrows():
                recommendation = {
                    'name': crystal_data.get('Name', 'Unknown Crystal'),
                    'chakra': crystal_data.get('Chakra_Association', 'Universal'),
                    'helps_with': crystal_data.get('Helps_With', 'General healing'),
                    'properties': crystal_data.get('Healing_Properties', 'Mystical energy'),
                    'color': crystal_data.get('Color', 'Rainbow'),
                    'usage': crystal_data.get('Usage_Tips', 'Hold with intention'),
                    'affirmation': crystal_data.get('Affirmation', 'I am aligned with healing energy')
                }
                recommendations.append(recommendation)
            
            logger.info(f"🌈 Found {len(recommendations)} crystals for {chakra_name} chakra")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error finding chakra crystals: {str(e)}")
            return []

    def get_database_stats(self) -> Dict[str, int]:
        """
        Get statistics about the crystal database.
        
        Returns:
            Dict[str, int]: Database statistics
        """
        try:
            stats = {
                'total_crystals': len(self.df),
                'unique_chakras': self.df['Chakra_Association'].nunique(),
                'unique_colors': self.df['Color'].nunique(),
                'crystals_with_affirmations': self.df['Affirmation'].notna().sum()
            }
            return stats
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {str(e)}")
            return {}
