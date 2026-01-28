"""
Multimodal processor for handling tables and charts in financial documents.
Supports OCR for tables and chart analysis for visual content.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from PIL import Image
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """Represents extracted table data."""
    content: str
    structure: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ChartData:
    """Represents extracted chart data."""
    description: str
    key_insights: List[str]
    data_points: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class MultimodalProcessor:
    """Processor for extracting and analyzing tables and charts from documents."""
    
    def __init__(self):
        self.table_patterns = [
            r'\b(?:table|schedule|exhibit)\s*\d+',
            r'\b(?:balance\s+sheet|income\s+statement|cash\s+flow)\b',
            r'\b(?:revenue|profit|loss|assets|liabilities)\b'
        ]
        
    def extract_tables_from_pdf(self, pdf_path: str) -> List[TableData]:
        """Extract tables from PDF using OCR and image processing."""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=300)
            tables = []
            
            for page_num, image in enumerate(images):
                # Convert PIL image to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detect table regions
                table_regions = self._detect_table_regions(cv_image)
                
                for region in table_regions:
                    table_data = self._extract_table_from_region(cv_image, region, page_num)
                    if table_data:
                        tables.append(table_data)
                        
            logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
            return []
    
    def _detect_table_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect table regions in an image using computer vision techniques."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines to get table grid
            table_grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours of table regions
            contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            table_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out small regions that are likely not tables
                if w > 100 and h > 50:
                    table_regions.append((x, y, w, h))
                    
            return table_regions
            
        except Exception as e:
            logger.error(f"Error detecting table regions: {e}")
            return []
    
    def _extract_table_from_region(self, image: np.ndarray, region: Tuple[int, int, int, int], page_num: int) -> Optional[TableData]:
        """Extract table data from a specific region using OCR."""
        try:
            x, y, w, h = region
            table_image = image[y:y+h, x:x+w]
            
            # Use Tesseract to extract text from table
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(table_image, config=custom_config)
            
            if not text.strip():
                return None
                
            # Clean and structure the table data
            cleaned_text = self._clean_table_text(text)
            structure = self._analyze_table_structure(cleaned_text)
            
            metadata = {
                'page_number': page_num,
                'region': region,
                'dimensions': (w, h),
                'ocr_confidence': pytesseract.image_to_data(table_image, output_type=pytesseract.Output.DICT)['conf']
            }
            
            return TableData(
                content=cleaned_text,
                structure=structure,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting table from region: {e}")
            return None
    
    def _clean_table_text(self, text: str) -> str:
        """Clean and normalize extracted table text."""
        # Remove extra whitespace and normalize formatting
        cleaned = re.sub(r'\s+', ' ', text.strip())
        cleaned = re.sub(r'\|+', '|', cleaned)  # Normalize separators
        
        # Handle common financial table patterns
        cleaned = re.sub(r'(\d),(\d)', r'\1\2', cleaned)  # Remove commas in numbers
        
        return cleaned
    
    def _analyze_table_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of extracted table data."""
        lines = text.split('\n')
        
        # Detect headers (usually first line or lines with specific patterns)
        headers = []
        data_rows = []
        
        for line in lines:
            if line.strip():
                # Simple heuristic: headers often contain words, data contains numbers
                if any(char.isdigit() for char in line):
                    data_rows.append(line)
                else:
                    headers.append(line)
        
        structure = {
            'headers': headers,
            'data_rows': len(data_rows),
            'columns': len(headers) if headers else 0,
            'has_headers': len(headers) > 0
        }
        
        return structure
    
    def extract_charts_from_pdf(self, pdf_path: str) -> List[ChartData]:
        """Extract and analyze charts from PDF documents."""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            charts = []
            
            for page_num, image in enumerate(images):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                chart_regions = self._detect_chart_regions(cv_image)
                
                for region in chart_regions:
                    chart_data = self._analyze_chart_region(cv_image, region, page_num)
                    if chart_data:
                        charts.append(chart_data)
                        
            logger.info(f"Extracted {len(charts)} charts from {pdf_path}")
            return charts
            
        except Exception as e:
            logger.error(f"Error extracting charts from {pdf_path}: {e}")
            return []
    
    def _detect_chart_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect chart regions in an image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection to find chart boundaries
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours that might be chart regions
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            chart_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter for chart-like regions (aspect ratio, size)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 3.0 and w > 150 and h > 100:
                    chart_regions.append((x, y, w, h))
                    
            return chart_regions
            
        except Exception as e:
            logger.error(f"Error detecting chart regions: {e}")
            return []
    
    def _analyze_chart_region(self, image: np.ndarray, region: Tuple[int, int, int, int], page_num: int) -> Optional[ChartData]:
        """Analyze a chart region to extract insights."""
        try:
            x, y, w, h = region
            chart_image = image[y:y+h, x:x+w]
            
            # Analyze chart type and extract basic information
            chart_type = self._detect_chart_type(chart_image)
            description = self._describe_chart(chart_image, chart_type)
            insights = self._extract_chart_insights(chart_image, chart_type)
            
            metadata = {
                'page_number': page_num,
                'region': region,
                'dimensions': (w, h),
                'chart_type': chart_type
            }
            
            return ChartData(
                description=description,
                key_insights=insights,
                data_points=[],  # Would require more sophisticated analysis
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing chart region: {e}")
            return None
    
    def _detect_chart_type(self, image: np.ndarray) -> str:
        """Detect the type of chart (bar, line, pie, etc.)."""
        try:
            # Simple heuristic-based chart type detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect lines (common in line charts and bar charts)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                # Count horizontal vs vertical lines
                horizontal_lines = 0
                vertical_lines = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y2 - y1) < abs(x2 - x1):  # Horizontal line
                        horizontal_lines += 1
                    else:  # Vertical line
                        vertical_lines += 1
                
                if vertical_lines > horizontal_lines:
                    return "bar_chart"
                else:
                    return "line_chart"
            
            # Check for circular patterns (pie charts)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=0, maxRadius=0)
            
            if circles is not None:
                return "pie_chart"
                
            return "unknown_chart"
            
        except Exception as e:
            logger.error(f"Error detecting chart type: {e}")
            return "unknown_chart"
    
    def _describe_chart(self, image: np.ndarray, chart_type: str) -> str:
        """Generate a textual description of the chart."""
        try:
            # Use OCR to extract any text labels from the chart
            text = pytesseract.image_to_string(image)
            
            description = f"Detected {chart_type.replace('_', ' ')} chart"
            if text.strip():
                description += f" with labels: {text.strip()}"
                
            return description
            
        except Exception as e:
            logger.error(f"Error describing chart: {e}")
            return f"Chart of type: {chart_type}"
    
    def _extract_chart_insights(self, image: np.ndarray, chart_type: str) -> List[str]:
        """Extract basic insights from chart analysis."""
        insights = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if chart_type == "bar_chart":
                # Analyze bar heights
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                bar_heights = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 10:  # Filter out noise
                        bar_heights.append(h)
                
                if bar_heights:
                    max_height = max(bar_heights)
                    min_height = min(bar_heights)
                    insights.append(f"Chart shows {len(bar_heights)} data points")
                    insights.append(f"Height range: {min_height} to {max_height} pixels")
                    
            elif chart_type == "line_chart":
                # Analyze line trends
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
                
                if lines is not None:
                    insights.append(f"Line chart with {len(lines)} detected line segments")
                    
            elif chart_type == "pie_chart":
                insights.append("Pie chart detected - represents proportional data")
                
        except Exception as e:
            logger.error(f"Error extracting chart insights: {e}")
            
        return insights
    
    def process_document_multimodal(self, pdf_path: str) -> Dict[str, Any]:
        """Process a document for both tables and charts."""
        try:
            logger.info(f"Processing multimodal content in {pdf_path}")
            
            tables = self.extract_tables_from_pdf(pdf_path)
            charts = self.extract_charts_from_pdf(pdf_path)
            
            result = {
                'tables': tables,
                'charts': charts,
                'summary': {
                    'total_tables': len(tables),
                    'total_charts': len(charts),
                    'document_path': pdf_path
                }
            }
            
            logger.info(f"Multimodal processing complete: {len(tables)} tables, {len(charts)} charts")
            return result
            
        except Exception as e:
            logger.error(f"Error in multimodal processing: {e}")
            return {'tables': [], 'charts': [], 'summary': {'error': str(e)}}