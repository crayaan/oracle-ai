import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, poppler_path=None, tesseract_cmd=None):
        """
        Initialize the PDF processor with optional paths to Poppler and Tesseract.
        
        Args:
            poppler_path (str, optional): Path to Poppler binaries
            tesseract_cmd (str, optional): Path to Tesseract executable
        """
        if poppler_path:
            os.environ['PATH'] += os.pathsep + poppler_path
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.supported_formats = {'.pdf'}

    def convert_pdf_to_text(self, pdf_path, output_dir=None, dpi=300):
        """
        Convert a PDF file to text using OCR.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str, optional): Directory to save the text file
            dpi (int): DPI for PDF to image conversion
            
        Returns:
            str: Path to the generated text file
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if pdf_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {pdf_path.suffix}")
        
        # Determine output directory and filename
        if output_dir is None:
            output_dir = pdf_path.parent / 'text_files'
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{pdf_path.stem}.txt"
        
        logger.info(f"Processing {pdf_path}")
        logger.info(f"Output will be saved to {output_file}")
        
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=dpi)
            
            # Process each page
            full_text = []
            for i, image in enumerate(images, 1):
                logger.info(f"Processing page {i}/{len(images)}")
                text = pytesseract.image_to_string(image)
                full_text.append(text)
            
            # Write the combined text to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(full_text))
            
            logger.info(f"Successfully converted {pdf_path} to text")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise

    def batch_convert(self, input_dir, output_dir=None, dpi=300):
        """
        Convert all PDFs in a directory to text files.
        
        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str, optional): Directory to save text files
            dpi (int): DPI for PDF to image conversion
            
        Returns:
            list: Paths to all generated text files
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = input_dir / 'text_files'
        
        output_files = []
        pdf_files = list(input_dir.glob('**/*.pdf'))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                output_file = self.convert_pdf_to_text(pdf_file, output_dir, dpi)
                output_files.append(output_file)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {str(e)}")
                continue
        
        return output_files

if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor()
    
    # Convert a single PDF
    # processor.convert_pdf_to_text("path/to/document.pdf", "path/to/output")
    
    # Convert all PDFs in a directory
    # processor.batch_convert("path/to/pdf/directory", "path/to/output") 