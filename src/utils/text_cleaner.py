import re
import logging
from typing import List, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    def __init__(self):
        """Initialize the TextCleaner with common cleaning patterns."""
        # Common patterns for cleaning
        self.patterns = {
            'multiple_newlines': re.compile(r'\n{3,}'),
            'multiple_spaces': re.compile(r' {2,}'),
            'page_numbers': re.compile(r'\b\d+\b(?=\s*$)'),
            'header_footer': re.compile(r'^.*(?:Page|Chapter|Section).*$', re.MULTILINE),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email': re.compile(r'\S+@\S+\.\S+'),
            'special_chars': re.compile(r'[^\w\s.,!?"\'-]'),
            'ellipsis': re.compile(r'\.{3,}'),
        }

    def clean_text(self, text: str, 
                  remove_urls: bool = True,
                  remove_emails: bool = True,
                  remove_special_chars: bool = True,
                  normalize_whitespace: bool = True,
                  remove_page_numbers: bool = True,
                  remove_headers_footers: bool = True,
                  normalize_quotes: bool = True) -> str:
        """
        Clean text using specified cleaning options.
        
        Args:
            text (str): Input text to clean
            remove_urls (bool): Remove URLs from text
            remove_emails (bool): Remove email addresses
            remove_special_chars (bool): Remove special characters
            normalize_whitespace (bool): Normalize whitespace and newlines
            remove_page_numbers (bool): Remove page numbers
            remove_headers_footers (bool): Remove headers and footers
            normalize_quotes (bool): Normalize different types of quotes
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return text

        # Store original length for logging
        original_length = len(text)
        
        try:
            # Remove URLs
            if remove_urls:
                text = self.patterns['urls'].sub('', text)
            
            # Remove email addresses
            if remove_emails:
                text = self.patterns['email'].sub('', text)
            
            # Remove page numbers
            if remove_page_numbers:
                text = self.patterns['page_numbers'].sub('', text)
            
            # Remove headers and footers
            if remove_headers_footers:
                text = self.patterns['header_footer'].sub('', text)
            
            # Normalize quotes
            if normalize_quotes:
                text = self._normalize_quotes(text)
            
            # Remove special characters
            if remove_special_chars:
                text = self.patterns['special_chars'].sub('', text)
            
            # Normalize whitespace
            if normalize_whitespace:
                # Replace multiple newlines with double newline
                text = self.patterns['multiple_newlines'].sub('\n\n', text)
                # Replace multiple spaces with single space
                text = self.patterns['multiple_spaces'].sub(' ', text)
                # Strip leading/trailing whitespace
                text = text.strip()
            
            # Log the cleaning results
            cleaned_length = len(text)
            reduction = ((original_length - cleaned_length) / original_length) * 100
            logger.info(f"Cleaned text: {cleaned_length} chars (reduced by {reduction:.1f}%)")
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text

    def clean_file(self, input_path: Union[str, Path], 
                  output_path: Optional[Union[str, Path]] = None,
                  **cleaning_options) -> str:
        """
        Clean text from a file and optionally save to a new file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file (optional)
            **cleaning_options: Options to pass to clean_text()
            
        Returns:
            str: Path to the output file
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean the text
            cleaned_text = self.clean_text(text, **cleaning_options)
            
            # Determine output path
            if output_path is None:
                output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
            else:
                output_path = Path(output_path)
                
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write cleaned text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
                
            logger.info(f"Cleaned text saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error processing file {input_path}: {str(e)}")
            raise

    def clean_files(self, input_dir: Union[str, Path], 
                   output_dir: Optional[Union[str, Path]] = None,
                   file_pattern: str = "*.txt",
                   **cleaning_options) -> List[str]:
        """
        Clean multiple text files in a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files (optional)
            file_pattern: Glob pattern for input files
            **cleaning_options: Options to pass to clean_text()
            
        Returns:
            list: Paths to all output files
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        if output_dir is None:
            output_dir = input_dir / 'cleaned'
        else:
            output_dir = Path(output_dir)
            
        output_files = []
        for input_file in input_dir.glob(file_pattern):
            try:
                output_file = self.clean_file(
                    input_file,
                    output_dir / input_file.name,
                    **cleaning_options
                )
                output_files.append(output_file)
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {str(e)}")
                continue
                
        return output_files

    def _normalize_quotes(self, text: str) -> str:
        """Normalize different types of quotes to standard quotes."""
        # Replace curly quotes with straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text

if __name__ == "__main__":
    # Example usage
    cleaner = TextCleaner()
    
    # Clean a single string
    # cleaned_text = cleaner.clean_text("Your text here...")
    
    # Clean a single file
    # cleaner.clean_file("path/to/input.txt", "path/to/output.txt")
    
    # Clean multiple files
    # cleaner.clean_files("path/to/input/dir", "path/to/output/dir") 