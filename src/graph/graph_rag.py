"""
GraphRAG Module

Implements knowledge graph functionality to map relationships between entities
in financial documents. Uses spaCy for entity extraction and LLM for relationship
identification, with NetworkX for graph construction and PyVis for visualization.
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import defaultdict
import re
import streamlit as st
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity extracted from documents"""
    text: str
    label: str
    confidence: float
    source_docs: List[str]
    page_numbers: List[int]
    
    def __hash__(self):
        return hash((self.text, self.label))


@dataclass
class Relationship:
    """Represents a relationship between two entities"""
    source: Entity
    target: Entity
    relationship_type: str
    confidence: float
    context: str
    source_docs: List[str]


class GraphRAG:
    """Knowledge graph builder for financial documents"""
    
    def __init__(self, llm):
        """
        Initialize GraphRAG system
        
        Args:
            llm: Language model for relationship extraction
        """
        self.llm = llm
        self.nlp = None
        self.graph = nx.Graph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        
        # Financial entity patterns
        self.financial_keywords = {
            'COMPANY': ['inc', 'corp', 'llc', 'ltd', 'plc', 'co', 'group'],
            'FINANCIAL_METRIC': ['revenue', 'profit', 'loss', 'expense', 'cost', 'income', 'earnings', 'margin'],
            'LOCATION': ['headquarters', 'office', 'facility', 'plant', 'warehouse', 'country', 'state', 'city'],
            'PERSON': ['ceo', 'cfo', 'president', 'director', 'executive', 'officer', 'manager']
        }
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            st.warning("⚠️ spaCy model not found. Entity extraction may be limited.")
    
    def extract_entities_from_documents(self, documents: List[Dict]) -> List[Entity]:
        """
        Extract entities from a list of documents
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            List of extracted entities
        """
        extracted_entities = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            source_file = metadata.get('source', 'Unknown')
            page_num = metadata.get('page', 0) + 1
            
            # Extract entities using spaCy
            if self.nlp:
                doc_nlp = self.nlp(content)
                
                for ent in doc_nlp.ents:
                    # Filter and enhance entities
                    entity = self._create_entity(
                        ent.text, ent.label_, ent.confidence if hasattr(ent, 'confidence') else 1.0,
                        source_file, page_num
                    )
                    if entity:
                        extracted_entities.append(entity)
            
            # Extract financial entities using patterns
            financial_entities = self._extract_financial_entities(content, source_file, page_num)
            extracted_entities.extend(financial_entities)
        
        # Deduplicate entities
        unique_entities = self._deduplicate_entities(extracted_entities)
        
        logger.info(f"Extracted {len(unique_entities)} unique entities from documents")
        return unique_entities
    
    def _create_entity(self, text: str, label: str, confidence: float, source_file: str, page_num: int) -> Optional[Entity]:
        """Create an entity with validation and enhancement"""
        # Clean entity text
        text = text.strip()
        if not text or len(text) < 2:
            return None
        
        # Filter out stop words and common noise
        if text.lower() in STOP_WORDS:
            return None
        
        # Enhance label for financial context
        enhanced_label = self._enhance_entity_label(text, label)
        
        # Create or update entity
        entity_key = f"{text.lower()}_{enhanced_label}"
        
        if entity_key in self.entities:
            # Update existing entity
            existing = self.entities[entity_key]
            existing.source_docs.append(source_file)
            existing.page_numbers.append(page_num)
            return existing
        else:
            # Create new entity
            entity = Entity(
                text=text,
                label=enhanced_label,
                confidence=confidence,
                source_docs=[source_file],
                page_numbers=[page_num]
            )
            self.entities[entity_key] = entity
            return entity
    
    def _enhance_entity_label(self, text: str, original_label: str) -> str:
        """Enhance entity label based on financial context"""
        text_lower = text.lower()
        
        # Check against financial keywords
        for category, keywords in self.financial_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        # Map common spaCy labels to financial categories
        label_mapping = {
            'ORG': 'COMPANY',
            'PERSON': 'PERSON',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'MONEY': 'FINANCIAL_METRIC',
            'PERCENT': 'FINANCIAL_METRIC',
            'DATE': 'TIME_PERIOD'
        }
        
        return label_mapping.get(original_label, original_label)
    
    def _extract_financial_entities(self, content: str, source_file: str, page_num: int) -> List[Entity]:
        """Extract financial entities using pattern matching"""
        entities = []
        
        # Extract company names (patterns like "Apple Inc.", "Microsoft Corporation")
        company_patterns = [
            r'\b([A-Z][a-zA-Z\s]+(?:Inc|Corp|LLC|Ltd|Plc|Group))\b',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:Inc|Corporation|Corp|LLC|Ltd)\b'
        ]
        
        for pattern in company_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entity = self._create_entity(
                    match.group(0), 'COMPANY', 0.8, source_file, page_num
                )
                if entity:
                    entities.append(entity)
        
        # Extract financial metrics
        metric_patterns = [
            r'\b(revenue|profit|loss|expense|cost|income|earnings|margin)\s*(?:was|is|are|were)?\s*\$?[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?\b',
            r'\b(\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?)\b'
        ]
        
        for pattern in metric_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entity = self._create_entity(
                    match.group(0), 'FINANCIAL_METRIC', 0.7, source_file, page_num
                )
                if entity:
                    entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities based on text and label"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relationships(self, entities: List[Entity], documents: List[Dict]) -> List[Relationship]:
        """
        Extract relationships between entities using LLM analysis
        
        Args:
            entities: List of entities to find relationships for
            documents: List of documents to analyze
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Group entities by document for efficient processing
        entity_groups = self._group_entities_by_document(entities)
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            source_file = doc.get('metadata', {}).get('source', 'Unknown')
            
            if source_file not in entity_groups:
                continue
            
            doc_entities = entity_groups[source_file]
            
            # Extract relationships within this document
            doc_relationships = self._extract_relationships_from_document(
                content, doc_entities, source_file
            )
            relationships.extend(doc_relationships)
        
        # Deduplicate relationships
        self.relationships = self._deduplicate_relationships(relationships)
        
        logger.info(f"Extracted {len(self.relationships)} unique relationships")
        return self.relationships
    
    def extract_relationships_fast(self, entities: List[Entity], documents: List[Dict]) -> List[Relationship]:
        """
        Fast relationship extraction using pattern matching instead of LLM
        
        Args:
            entities: List of entities to find relationships for
            documents: List of documents to analyze
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Create entity lookup for fast access
        entity_lookup = {entity.text.lower(): entity for entity in entities}
        
        # Pattern-based relationship extraction
        relationship_patterns = [
            # Company relationships
            (r'(\w+)\s+(?:supplies|provides)\s+to\s+(\w+)', 'SUPPLIES_TO'),
            (r'(\w+)\s+(?:competes|competes with)\s+(\w+)', 'COMPETES_WITH'),
            (r'(\w+)\s+(?:headquartered|located)\s+in\s+(\w+)', 'HEADQUARTERED_IN'),
            (r'(\w+)\s+(?:employs|hires)\s+(\w+)', 'EMPLOYS'),
            (r'(\w+)\s+(?:owns|acquired)\s+(\w+)', 'OWNS'),
            
            # Financial relationships
            (r'(\w+)\s+(?:revenue|sales)\s+of\s+(\$[\d,]+)', 'HAS_REVENUE'),
            (r'(\w+)\s+(?:profit|earnings)\s+of\s+(\$[\d,]+)', 'HAS_PROFIT'),
            (r'(\w+)\s+(?:loss)\s+of\s+(\$[\d,]+)', 'HAS_LOSS'),
            (r'(\w+)\s+(?:expense|cost)\s+of\s+(\$[\d,]+)', 'HAS_EXPENSE'),
        ]
        
        for doc in documents:
            content = doc.get('content', '')
            source_file = doc.get('metadata', {}).get('source', 'Unknown')
            
            # Apply each pattern to extract relationships
            for pattern, rel_type in relationship_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    entity1_text = match.group(1)
                    entity2_text = match.group(2)
                    
                    # Find entities in our entity list
                    source_entity = entity_lookup.get(entity1_text.lower())
                    target_entity = entity_lookup.get(entity2_text.lower())
                    
                    if source_entity and target_entity:
                        relationship = Relationship(
                            source=source_entity,
                            target=target_entity,
                            relationship_type=rel_type,
                            confidence=0.7,  # Pattern-based confidence
                            context=match.group(0),
                            source_docs=[source_file]
                        )
                        relationships.append(relationship)
        
        # Deduplicate relationships
        self.relationships = self._deduplicate_relationships(relationships)
        
        logger.info(f"Fast extraction: Found {len(self.relationships)} relationships")
        return self.relationships
    
    def _group_entities_by_document(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """Group entities by their source documents"""
        groups = defaultdict(list)
        for entity in entities:
            for source in entity.source_docs:
                groups[source].append(entity)
        return dict(groups)
    
    def _extract_relationships_from_document(self, content: str, entities: List[Entity], source_file: str) -> List[Relationship]:
        """Extract relationships from a single document using LLM"""
        if len(entities) < 2:
            return []
        
        # Create entity pairs for relationship extraction
        relationships = []
        
        # Use LLM to identify relationships in chunks of content
        content_chunks = self._split_content_for_analysis(content)
        
        for chunk in content_chunks:
            chunk_relationships = self._analyze_chunk_for_relationships(chunk, entities, source_file)
            relationships.extend(chunk_relationships)
        
        return relationships
    
    def _split_content_for_analysis(self, content: str, max_chunk_size: int = 2000) -> List[str]:
        """Split content into manageable chunks for LLM analysis"""
        if len(content) <= max_chunk_size:
            return [content]
        
        # Split by sentences to maintain context
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
            
            current_chunk += sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _analyze_chunk_for_relationships(self, chunk: str, entities: List[Entity], source_file: str) -> List[Relationship]:
        """Analyze a content chunk for relationships between entities"""
        relationships = []
        
        # Create entity name list for prompt
        entity_names = [entity.text for entity in entities]
        
        # LLM prompt for relationship extraction
        prompt = f"""
        Analyze the following text for relationships between the specified entities:

        Text: "{chunk}"

        Entities: {', '.join(entity_names)}

        Identify relationships between these entities. For each relationship, provide:
        1. Source entity
        2. Target entity  
        3. Relationship type (e.g., "SUPPLIES_TO", "COMPETES_WITH", "HEADQUARTERED_IN", "EMPLOYS", "OWNS")
        4. Brief context description

        Return the results in JSON format with the following structure:
        [
          {{
            "source": "entity_name",
            "target": "entity_name", 
            "relationship_type": "RELATIONSHIP_TYPE",
            "context": "brief description"
          }}
        ]

        Only include relationships that are explicitly mentioned or strongly implied in the text.
        """
        
        try:
            # Call LLM for relationship extraction
            response = self.llm.invoke(prompt)
            relationships_data = self._parse_relationship_response(response)
            
            # Convert to Relationship objects
            for rel_data in relationships_data:
                source_entity = self._find_entity_by_text(rel_data['source'], entities)
                target_entity = self._find_entity_by_text(rel_data['target'], entities)
                
                if source_entity and target_entity:
                    relationship = Relationship(
                        source=source_entity,
                        target=target_entity,
                        relationship_type=rel_data['relationship_type'],
                        confidence=0.8,  # Default confidence
                        context=rel_data['context'],
                        source_docs=[source_file]
                    )
                    relationships.append(relationship)
        
        except Exception as e:
            logger.error(f"Error extracting relationships from chunk: {e}")
        
        return relationships
    
    def _parse_relationship_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract relationship data"""
        relationships = []
        
        # Simple JSON parsing - in production, use more robust parsing
        try:
            import json
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                relationships = json.loads(json_str)
        except:
            logger.warning("Failed to parse relationship JSON response")
        
        return relationships
    
    def _find_entity_by_text(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find entity by text (case-insensitive)"""
        text_lower = text.lower().strip()
        for entity in entities:
            if entity.text.lower().strip() == text_lower:
                return entity
        return None
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Deduplicate relationships based on source, target, and type"""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            key = (rel.source.text.lower(), rel.target.text.lower(), rel.relationship_type.lower())
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    def build_knowledge_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """
        Build the knowledge graph from entities and relationships
        
        Args:
            entities: List of entities
            relationships: List of relationships
        """
        self.graph = nx.Graph()
        
        # Add nodes (entities)
        for entity in entities:
            self.graph.add_node(
                entity.text,
                label=entity.label,
                confidence=entity.confidence,
                source_docs=entity.source_docs,
                page_numbers=entity.page_numbers
            )
        
        # Add edges (relationships)
        for relationship in relationships:
            self.graph.add_edge(
                relationship.source.text,
                relationship.target.text,
                relationship_type=relationship.relationship_type,
                confidence=relationship.confidence,
                context=relationship.context,
                source_docs=relationship.source_docs
            )
        
        logger.info(f"Built knowledge graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if not self.graph:
            return {}
        
        stats = {
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(self.graph.edges),
            'node_types': defaultdict(int),
            'edge_types': defaultdict(int),
            'connected_components': nx.number_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if self.graph.nodes else 0,
            'most_connected': [],
            'top_relationships': []
        }
        
        # Count node types
        for node, attrs in self.graph.nodes(data=True):
            stats['node_types'][attrs.get('label', 'Unknown')] += 1
        
        # Count edge types
        for u, v, attrs in self.graph.edges(data=True):
            stats['edge_types'][attrs.get('relationship_type', 'Unknown')] += 1
        
        # Most connected nodes
        degrees = dict(self.graph.degree())
        stats['most_connected'] = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top relationships
        rel_counts = defaultdict(int)
        for u, v, attrs in self.graph.edges(data=True):
            rel_type = attrs.get('relationship_type', 'Unknown')
            rel_counts[rel_type] += 1
        
        stats['top_relationships'] = sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return stats
    
    def export_graph_data(self, filename: str = None) -> str:
        """
        Export graph data to various formats
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if not self.graph:
            logger.warning("No graph to export")
            return ""
        
        if filename is None:
            filename = f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create export directory
        export_dir = Path("data/graphs")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to GraphML
        graphml_path = export_dir / f"{filename}.graphml"
        nx.write_graphml(self.graph, str(graphml_path))
        
        # Export to GEXF (Gephi format)
        gexf_path = export_dir / f"{filename}.gexf"
        nx.write_gexf(self.graph, str(gexf_path))
        
        # Export node/edge data to CSV
        nodes_df = nx.to_pandas_edgelist(self.graph)
        csv_path = export_dir / f"{filename}_edges.csv"
        nodes_df.to_csv(str(csv_path), index=False)
        
        logger.info(f"Graph exported to: {graphml_path}, {gexf_path}, {csv_path}")
        return str(graphml_path)


def create_graph_visualization(graph_rag: GraphRAG):
    """
    Create interactive graph visualization using PyVis
    
    Args:
        graph_rag: GraphRAG instance with built graph
    """
    if not graph_rag.graph or len(graph_rag.graph.nodes) == 0:
        st.info("""
        **No knowledge graph available yet.** 
        
        To build a knowledge graph, follow these steps:
        
        1. **Select documents** from the sidebar and click "Activate Selection"
        2. **Click "Build Knowledge Graph"** below to process the documents
        3. **Wait for processing** - the system will extract entities and relationships
        4. **View the interactive graph** once processing is complete
        
        **What happens during processing:**
        - Entity extraction using spaCy NLP
        - Relationship identification using LLM analysis
        - Graph construction and visualization
        
        **Example entities found:**
        - Companies: Apple Inc., Tesla Motors
        - People: Executives, key personnel
        - Financial Metrics: Revenue, profit, expenses
        - Locations: Headquarters, facilities
        """)
        return
    
    st.subheader("🕸️ Knowledge Graph Visualization")
    
    st.write("""
    **What is this?**
    The Knowledge Graph visualizes relationships between entities extracted from your financial documents.
    It helps you understand connections between companies, people, locations, and financial metrics.
    
    **How it works:**
    1. **Entity Extraction**: Uses spaCy and pattern matching to identify key entities
    2. **Relationship Analysis**: LLM analyzes text to find connections between entities
    3. **Graph Building**: Creates a network graph showing entity relationships
    4. **Interactive Visualization**: Displays the graph with clickable nodes and edges
    
    **Entity Types:**
    - 🔴 **Companies**: Organizations, corporations, business entities
    - 🔵 **People**: Executives, employees, key personnel
    - 🟢 **Locations**: Headquarters, offices, facilities, geographic locations
    - 🟡 **Financial Metrics**: Revenue, profit, expenses, financial data points
    - 🟣 **Time Periods**: Dates, fiscal periods, time-related entities
    
    **Relationship Types:**
    - **SUPPLIES_TO**: Supplier-customer relationships
    - **COMPETES_WITH**: Competitive relationships
    - **HEADQUARTERED_IN**: Location relationships
    - **EMPLOYS**: Employment relationships
    - **OWNS**: Ownership relationships
    """)
    
    # Graph statistics
    stats = graph_rag.get_graph_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entities", stats['total_nodes'])
    
    with col2:
        st.metric("Relationships", stats['total_edges'])
    
    with col3:
        st.metric("Connected Components", stats['connected_components'])
    
    with col4:
        st.metric("Avg. Degree", f"{stats['avg_degree']:.1f}")
    
    # Entity type distribution
    st.subheader("📊 Entity Distribution")
    entity_types = dict(stats['node_types'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Entity Types:**")
        for entity_type, count in entity_types.items():
            st.write(f"- {entity_type}: {count}")
    
    with col2:
        st.write("**Top Relationships:**")
        for rel_type, count in stats['top_relationships'][:5]:
            st.write(f"- {rel_type}: {count}")
    
    # Interactive graph visualization
    st.subheader("🌐 Interactive Graph")
    
    # PyVis visualization
    try:
        from pyvis.network import Network
        
        # Create PyVis network
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            notebook=False
        )
        
        # Set physics options
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 100,
              "fit": true
            },
            "repulsion": {
              "nodeDistance": 150,
              "centralGravity": 0.1
            }
          },
          "nodes": {
            "shape": "dot",
            "size": 15,
            "font": {
              "size": 12
            }
          },
          "edges": {
            "width": 2,
            "arrows": "to"
          }
        }
        """)
        
        # Add nodes with colors based on entity type
        color_map = {
            'COMPANY': '#e74c3c',
            'PERSON': '#3498db',
            'LOCATION': '#2ecc71',
            'FINANCIAL_METRIC': '#f1c40f',
            'TIME_PERIOD': '#9b59b6'
        }
        
        for node, attrs in graph_rag.graph.nodes(data=True):
            node_color = color_map.get(attrs.get('label', 'Unknown'), '#95a5a6')
            title = f"{node}<br>Type: {attrs.get('label', 'Unknown')}<br>Sources: {', '.join(attrs.get('source_docs', []))}"
            
            net.add_node(
                node,
                label=node,
                title=title,
                color=node_color,
                size=min(50, 10 + attrs.get('confidence', 1.0) * 40)
            )
        
        # Add edges
        for u, v, attrs in graph_rag.graph.edges(data=True):
            edge_title = f"{attrs.get('relationship_type', 'RELATIONSHIP')}<br>{attrs.get('context', '')}"
            
            net.add_edge(
                u, v,
                title=edge_title,
                width=2 + attrs.get('confidence', 1.0) * 3,
                color='#bdc3c7'
            )
        
        # Generate HTML
        html_path = f"data/graphs/graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        Path("data/graphs").mkdir(parents=True, exist_ok=True)
        net.save_graph(html_path)
        
        # Display in Streamlit
        st.components.v1.html(open(html_path, 'r').read(), height=650, scrolling=True)
        
    except ImportError:
        st.warning("PyVis not available. Install with: pip install pyvis")
        
        # Fallback: show graph info
        st.write("Graph nodes:", list(graph_rag.graph.nodes())[:10], "...")
        st.write("Graph edges:", list(graph_rag.graph.edges())[:10], "...")
    
    # Export options
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Graph Data", type="primary"):
            filepath = graph_rag.export_graph_data()
            if filepath:
                st.success(f"Graph exported to: {filepath}")
            else:
                st.error("Failed to export graph")
    
    with col2:
        if st.button("🔄 Rebuild Graph", type="secondary"):
            st.info("Graph will be rebuilt on next document processing")
    
    with col3:
        if st.button("🗑️ Clear Graph", type="secondary"):
            graph_rag.graph.clear()
            graph_rag.entities.clear()
            graph_rag.relationships.clear()
            st.rerun()