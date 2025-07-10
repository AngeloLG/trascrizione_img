import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import logging
import sys
import os

from src.file_handler import validate_image_file, save_transcription, load_prompt_file, SUPPORTED_IMAGE_EXTENSIONS

# Suppress most logging output during tests for clarity, unless a test specifically needs it
logging.disable(logging.CRITICAL) # Disable all logging less severe than CRITICAL

class TestFileHandler(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.test_dir = Path("test_data_temp") # Temporary directory for test artifacts
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up after test methods."""
        for item in self.test_dir.iterdir():
            item.unlink(missing_ok=True) # missing_ok for safety if file already gone
        if self.test_dir.exists():
            try:
                self.test_dir.rmdir()
            except OSError: # If not empty or other issues, don't fail test run
                pass

    def test_validate_image_file_valid_jpg(self):
        """Test validate_image_file with a valid .jpg file."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.suffix = '.jpg'
        mock_path.name = 'test_image.jpg'
        self.assertTrue(validate_image_file(mock_path))

    def test_validate_image_file_valid_jpeg(self):
        """Test validate_image_file with a valid .jpeg file."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.suffix = '.jpeg'
        mock_path.name = 'test_image.jpeg'
        self.assertTrue(validate_image_file(mock_path))

    def test_validate_image_file_invalid_extension(self):
        """Test validate_image_file with an invalid file extension."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.suffix = '.png'
        mock_path.name = 'test_image.png'
        self.assertFalse(validate_image_file(mock_path))

    def test_validate_image_file_not_exists(self):
        """Test validate_image_file with a non-existent file."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mock_path.is_file.return_value = True # Should not matter if not exists
        mock_path.suffix = '.jpg'
        mock_path.name = 'non_existent.jpg'
        self.assertFalse(validate_image_file(mock_path))

    def test_validate_image_file_is_not_file(self):
        """Test validate_image_file with a path that is not a file (e.g., a directory)."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = False
        mock_path.suffix = '.jpg'
        mock_path.name = 'a_directory.jpg'
        self.assertFalse(validate_image_file(mock_path))
    
    def test_validate_image_file_invalid_path_type(self):
        """Test validate_image_file with a non-Path object."""
        self.assertFalse(validate_image_file("not_a_path_object.jpg"))

    @patch('src.file_handler.Path.mkdir')
    @patch("builtins.open", new_callable=mock_open)
    def test_save_transcription_success_no_output_dir(self, mock_file_open, mock_mkdir):
        """Test save_transcription successfully saves when no output_dir is specified."""
        input_image_path = Path("dummy_input/image.jpg")
        transcription_text = "This is a test transcription."
        
        expected_output_path = input_image_path.parent / (input_image_path.stem + ".txt")
        
        result_path = save_transcription(transcription_text, input_image_path)

        mock_file_open.assert_called_once_with(expected_output_path, "w", encoding='utf-8')
        mock_file_open().write.assert_called_once_with(transcription_text)
        mock_mkdir.assert_not_called() # Should not be called if output_dir is None
        self.assertEqual(result_path, expected_output_path)

    @patch('src.file_handler.Path.mkdir')
    @patch("builtins.open", new_callable=mock_open)
    def test_save_transcription_success_with_output_dir(self, mock_file_open, mock_mkdir):
        """Test save_transcription successfully saves when output_dir is specified."""
        input_image_path = Path("dummy_input/image.jpg")
        output_directory = Path("custom_output")
        transcription_text = "Another test transcription."
        
        expected_output_path = output_directory / (input_image_path.stem + ".txt")
        
        result_path = save_transcription(transcription_text, input_image_path, output_directory)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file_open.assert_called_once_with(expected_output_path, "w", encoding='utf-8')
        mock_file_open().write.assert_called_once_with(transcription_text)
        self.assertEqual(result_path, expected_output_path)

    @patch('src.file_handler.Path.mkdir', side_effect=PermissionError("Test permission error"))
    def test_save_transcription_mkdir_permission_error(self, mock_mkdir):
        """Test save_transcription handles PermissionError during output_dir.mkdir."""
        input_image_path = Path("dummy_input/image.jpg")
        output_directory = Path("restricted_output")
        transcription_text = "Transcription won't save."
        
        result_path = save_transcription(transcription_text, input_image_path, output_directory)
        self.assertIsNone(result_path)
        mock_mkdir.assert_called_once()

    @patch('src.file_handler.Path.mkdir') # Mock mkdir to prevent actual creation
    @patch("builtins.open", new_callable=mock_open) # Mock open to prevent file writing
    def test_save_transcription_invalid_input_image_path_type(self, mock_file_open, mock_mkdir):
        """Test save_transcription with invalid input_image_path type."""
        self.assertIsNone(save_transcription("text", "not_a_path.jpg"))
        mock_mkdir.assert_not_called()

    @patch('src.file_handler.Path.mkdir') # Mock mkdir to prevent actual creation
    @patch("builtins.open", new_callable=mock_open) # Mock open to prevent file writing
    def test_save_transcription_invalid_output_dir_type(self, mock_file_open, mock_mkdir):
        """Test save_transcription with invalid output_directory type."""
        input_image_path = Path("dummy_input/image.jpg")
        self.assertIsNone(save_transcription("text", input_image_path, "not_a_path"))
        # mkdir might be called if output_directory is not None, but Path conversion will fail if it's not a Path
        # The type check is done before mkdir is called with the Path object.
        mock_mkdir.assert_not_called() # Because the type check for output_dir is first

    @patch('src.file_handler.Path.mkdir') # Keep this to avoid actual dir creation
    @patch("builtins.open", side_effect=IOError("Test IO error while opening file"))
    def test_save_transcription_io_error_on_write(self, mock_open_call, mock_mkdir):
        """Test save_transcription handles IOError during file write."""
        input_image_path = Path("dummy_input/image.jpg")
        output_directory = Path("output")
        transcription_text = "This will fail."
        
        result_path = save_transcription(transcription_text, input_image_path, output_directory)
        self.assertIsNone(result_path)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        expected_save_path = output_directory / (input_image_path.stem + ".txt")
        mock_open_call.assert_called_once_with(expected_save_path, "w", encoding='utf-8')

    @patch("src.file_handler.Path.is_file", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="Test prompt content.")
    def test_load_prompt_file_success(self, mock_file_open, mock_is_file):
        """Test load_prompt_file successfully loads content."""
        prompt_path = Path("test_prompt.txt")
        expected_content = "Test prompt content."
        
        content = load_prompt_file(prompt_path)
        
        mock_is_file.assert_called_once_with()
        mock_file_open.assert_called_once_with(prompt_path, "r", encoding='utf-8')
        self.assertEqual(content, expected_content)

    def test_load_prompt_file_not_is_file(self):
        """Test load_prompt_file when path is not a file."""
        mock_path = MagicMock(spec=Path)
        mock_path.is_file.return_value = False
        mock_path.name = "not_a_file.txt"
        self.assertIsNone(load_prompt_file(mock_path))
        mock_path.is_file.assert_called_once()

    @patch("src.file_handler.Path.is_file", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="") # Empty file
    def test_load_prompt_file_empty_content(self, mock_file_open, mock_is_file):
        """Test load_prompt_file with an empty prompt file."""
        prompt_path = Path("empty_prompt.txt")
        # Assuming empty content is acceptable and returns empty string (stripped)
        self.assertEqual(load_prompt_file(prompt_path), "")

    @patch("src.file_handler.Path.is_file", return_value=True)
    @patch("builtins.open", side_effect=IOError("Test IO Error when opening"))
    def test_load_prompt_file_io_error(self, mock_open_call, mock_is_file):
        """Test load_prompt_file handles IOError."""
        prompt_path = Path("error_prompt.txt")
        self.assertIsNone(load_prompt_file(prompt_path))
        mock_is_file.assert_called_once_with()
        mock_open_call.assert_called_once_with(prompt_path, "r", encoding='utf-8')

    @patch("src.file_handler.Path.is_file", return_value=True)
    @patch("builtins.open", side_effect=UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte'))
    def test_load_prompt_file_unicode_decode_error(self, mock_open_call, mock_is_file):
        """Test load_prompt_file handles UnicodeDecodeError."""
        prompt_path = Path("decode_error_prompt.txt")
        self.assertIsNone(load_prompt_file(prompt_path))
        mock_is_file.assert_called_once_with()
        mock_open_call.assert_called_once_with(prompt_path, "r", encoding='utf-8')
        
    def test_load_prompt_file_invalid_path_type(self):
        """Test load_prompt_file with a non-Path object."""
        self.assertIsNone(load_prompt_file("not_a_path.txt"))

if __name__ == '__main__':
    unittest.main() 