import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import os
import logging
import sys

from src.llm_handler import configure_openai_api, get_transcription_from_llm, image_to_base64, OPENAI_API_CONFIGURED, client as openai_client_global
from openai import AuthenticationError, RateLimitError, NotFoundError, BadRequestError, APIError # Import specific errors

# Suppress most logging output during tests for clarity
logging.disable(logging.CRITICAL)

class TestLLMHandler(unittest.TestCase):

    def setUp(self):
        """Reset global configuration state before each test."""
        import src.llm_handler as llm_handler_module_for_test # Import for modifying globals
        llm_handler_module_for_test.OPENAI_API_CONFIGURED = False
        llm_handler_module_for_test.client = None

        self.test_image_path = Path("test_data_temp/test_image.jpg")
        self.test_dir = Path("test_data_temp")
        self.test_dir.mkdir(exist_ok=True)
        try:
            with open(self.test_image_path, "w") as f:
                f.write("dummy image data")
        except IOError:
            pass

    def tearDown(self):
        """Clean up created files."""
        if self.test_image_path.exists():
            self.test_image_path.unlink()
        if self.test_dir.exists():
            try:
                 self.test_dir.rmdir()
            except OSError: 
                pass

    @patch('src.llm_handler.os.getenv')
    @patch('src.llm_handler.Path.exists')
    @patch('src.llm_handler.load_dotenv')
    @patch('src.llm_handler.OpenAI') 
    def test_configure_openai_api_success_from_env(self, mock_openai_constructor, mock_load_dotenv, mock_path_exists, mock_os_getenv):
        """Test successful API configuration from environment variable."""
        mock_os_getenv.return_value = "fake_api_key_from_env"
        mock_path_exists.return_value = False 
        mock_openai_instance = MagicMock()
        mock_openai_constructor.return_value = mock_openai_instance
        
        import src.llm_handler as llm_handler_module_for_test
        self.assertTrue(configure_openai_api())
        self.assertTrue(llm_handler_module_for_test.OPENAI_API_CONFIGURED)
        self.assertIsNotNone(llm_handler_module_for_test.client)
        mock_os_getenv.assert_called_once_with("OPENAI_API_KEY")
        mock_load_dotenv.assert_not_called()
        mock_openai_constructor.assert_called_once_with(api_key="fake_api_key_from_env")

    @patch('src.llm_handler.os.getenv')
    @patch('src.llm_handler.Path.exists')
    @patch('src.llm_handler.load_dotenv')
    @patch('src.llm_handler.OpenAI')
    def test_configure_openai_api_success_from_dotenv(self, mock_openai_constructor, mock_load_dotenv, mock_path_exists, mock_os_getenv):
        """Test successful API configuration from .env file."""
        mock_path_exists.return_value = True 
        mock_os_getenv.return_value = "fake_api_key_from_dotenv" 
        mock_openai_instance = MagicMock()
        mock_openai_constructor.return_value = mock_openai_instance

        import src.llm_handler as llm_handler_module_for_test
        self.assertTrue(configure_openai_api())
        self.assertTrue(llm_handler_module_for_test.OPENAI_API_CONFIGURED)
        mock_load_dotenv.assert_called_once()
        mock_os_getenv.assert_called_with("OPENAI_API_KEY") 
        mock_openai_constructor.assert_called_once_with(api_key="fake_api_key_from_dotenv")

    @patch('src.llm_handler.os.getenv', return_value=None)
    @patch('src.llm_handler.Path.exists', return_value=False)
    def test_configure_openai_api_no_key(self, mock_path_exists, mock_os_getenv):
        """Test API configuration failure when no API key is found."""
        import src.llm_handler as llm_handler_module_for_test
        self.assertFalse(configure_openai_api())
        self.assertFalse(llm_handler_module_for_test.OPENAI_API_CONFIGURED)
        self.assertIsNone(llm_handler_module_for_test.client)

    @patch('src.llm_handler.os.getenv', return_value="fake_key")
    @patch('src.llm_handler.Path.exists', return_value=False)
    @patch('src.llm_handler.OpenAI', side_effect=Exception("Connection failed"))
    def test_configure_openai_api_client_instantiation_fails(self, mock_openai_constructor, mock_path_exists, mock_os_getenv):
        """Test API configuration failure when OpenAI client instantiation fails."""
        import src.llm_handler as llm_handler_module_for_test
        self.assertFalse(configure_openai_api())
        self.assertFalse(llm_handler_module_for_test.OPENAI_API_CONFIGURED)
        self.assertIsNone(llm_handler_module_for_test.client)
        mock_openai_constructor.assert_called_once_with(api_key="fake_key")

    @patch('src.llm_handler.Image.open')
    @patch('builtins.open', new_callable=mock_open, read_data=b'imagedata')
    def test_image_to_base64_success_jpg(self, mock_file_open, mock_image_open):
        """Test successful image to base64 conversion for JPG."""
        mock_img_instance = MagicMock()
        mock_img_instance.format = 'JPEG'
        mock_image_open.return_value.__enter__.return_value = mock_img_instance
        
        base64_str = image_to_base64(self.test_image_path)
        self.assertIsNotNone(base64_str)
        self.assertTrue(base64_str.startswith("data:image/jpeg;base64,"))
        mock_image_open.assert_called_once_with(self.test_image_path)
        mock_file_open.assert_called_once_with(self.test_image_path, "rb")

    @patch('src.llm_handler.Image.open')
    def test_image_to_base64_success_png_guessed_from_suffix(self, mock_image_open):
        """Test image_to_base64 with format guessed from .png suffix."""
        mock_img_instance = MagicMock()
        mock_img_instance.format = None 
        mock_image_open.return_value.__enter__.return_value = mock_img_instance
        
        png_image_path = self.test_dir / "test_image.png"
        try:
            # 1. Create a real dummy file first so Path.is_file() passes in the tested function
            with open(png_image_path, "wb") as f:
                f.write(b"dummy png data for is_file check") 
            
            # 2. Now, when image_to_base64 is called, mock builtins.open for its internal read operation
            with patch('builtins.open', new_callable=mock_open, read_data=b'dummy png data for base64 encoding') as mock_builtin_open_for_read:
                base64_str = image_to_base64(png_image_path)
            
            self.assertIsNotNone(base64_str, "base64_str should not be None")
            self.assertTrue(base64_str.startswith("data:image/png;base64,"))
            mock_image_open.assert_called_once_with(png_image_path)
            # Ensure the builtins.open mock (for reading) was called as expected by image_to_base64
            mock_builtin_open_for_read.assert_called_once_with(png_image_path, "rb")
        finally:
            if png_image_path.exists(): 
                png_image_path.unlink()

    @patch('src.llm_handler.Path.is_file', return_value=False)
    def test_image_to_base64_file_not_found_early_exit(self, mock_is_file):
        """Test image_to_base64 when image file does not exist (is_file check)."""
        self.assertIsNone(image_to_base64(Path("non_existent_image.jpg")))
        mock_is_file.assert_called_once()

    @patch('src.llm_handler.Image.open', side_effect=IOError("Cannot open image"))
    def test_image_to_base64_image_open_io_error(self, mock_image_open):
        """Test image_to_base64 handles IOError when Image.open fails."""
        with patch('src.llm_handler.Path.is_file', return_value=True):
             self.assertIsNone(image_to_base64(self.test_image_path))
        mock_image_open.assert_called_once_with(self.test_image_path)

    @patch('src.llm_handler.configure_openai_api', return_value=True) 
    @patch('src.llm_handler.image_to_base64', return_value="fake_base64_string")
    def test_get_transcription_from_llm_success(self, mock_img_to_base64, mock_configure_api):
        """Test successful transcription from LLM."""
        import src.llm_handler as llm_handler_module_for_test
        llm_handler_module_for_test.OPENAI_API_CONFIGURED = True
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = " Test transcription "
        mock_client.chat.completions.create.return_value = mock_response
        llm_handler_module_for_test.client = mock_client

        prompt = "Transcribe this image."
        transcription = get_transcription_from_llm(self.test_image_path, prompt)
        
        self.assertEqual(transcription, "Test transcription")
        mock_img_to_base64.assert_called_once_with(self.test_image_path)
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs['model'], "gpt-4o")
        self.assertEqual(kwargs['messages'][0]['content'][0]['text'], prompt)
        self.assertEqual(kwargs['messages'][0]['content'][1]['image_url']['url'], "fake_base64_string")

    @patch('src.llm_handler.configure_openai_api', return_value=False) 
    def test_get_transcription_from_llm_api_config_fails(self, mock_configure_api):
        """Test get_transcription_from_llm when API configuration fails initially."""
        import src.llm_handler as llm_handler_module_for_test
        llm_handler_module_for_test.OPENAI_API_CONFIGURED = False 
        llm_handler_module_for_test.client = None

        transcription = get_transcription_from_llm(self.test_image_path, "prompt")
        self.assertIsNone(transcription)
        mock_configure_api.assert_called_once() 

    @patch('src.llm_handler.configure_openai_api', return_value=True)
    @patch('src.llm_handler.image_to_base64', return_value=None) 
    def test_get_transcription_from_llm_image_conversion_fails(self, mock_img_to_base64, mock_configure_api):
        """Test get_transcription_from_llm when image_to_base64 fails."""
        import src.llm_handler as llm_handler_module_for_test
        llm_handler_module_for_test.OPENAI_API_CONFIGURED = True
        llm_handler_module_for_test.client = MagicMock() 

        transcription = get_transcription_from_llm(self.test_image_path, "prompt")
        self.assertIsNone(transcription)
        mock_img_to_base64.assert_called_once_with(self.test_image_path)

    @patch('src.llm_handler.configure_openai_api', return_value=True)
    @patch('src.llm_handler.image_to_base64', return_value="fake_base64_string")
    def _run_openai_api_error_test(self, api_error_to_raise, mock_img_to_base64, mock_configure_api):
        import src.llm_handler as llm_handler_module_for_test
        llm_handler_module_for_test.OPENAI_API_CONFIGURED = True
        mock_client = MagicMock()
        if isinstance(api_error_to_raise, BadRequestError):
            error_response_mock = MagicMock()
            error_response_mock.text = "Invalid image format or size"
            api_error_to_raise.response = error_response_mock
        mock_client.chat.completions.create.side_effect = api_error_to_raise
        llm_handler_module_for_test.client = mock_client

        transcription = get_transcription_from_llm(self.test_image_path, "prompt")
        self.assertIsNone(transcription)
        mock_client.chat.completions.create.assert_called_once()

    def test_get_transcription_from_llm_handles_authentication_error(self):
        """Test handling of AuthenticationError from OpenAI API."""
        self._run_openai_api_error_test(AuthenticationError("Invalid API Key", response=MagicMock(), body=None))
    
    def test_get_transcription_from_llm_handles_rate_limit_error(self):
        """Test handling of RateLimitError from OpenAI API."""
        self._run_openai_api_error_test(RateLimitError("Rate limit exceeded", response=MagicMock(), body=None))

    def test_get_transcription_from_llm_handles_not_found_error(self):
        """Test handling of NotFoundError (e.g., model not found) from OpenAI API."""
        self._run_openai_api_error_test(NotFoundError("Model not found", response=MagicMock(), body=None))

    def test_get_transcription_from_llm_handles_bad_request_error(self):
        """Test handling of BadRequestError (e.g., invalid image) from OpenAI API."""
        self._run_openai_api_error_test(BadRequestError("Bad request", response=MagicMock(), body=None))

    def test_get_transcription_from_llm_handles_generic_api_error(self):
        """Test handling of a generic APIError from OpenAI API."""
        self._run_openai_api_error_test(APIError("Generic API error", request=MagicMock()))

    @patch('src.llm_handler.configure_openai_api', return_value=True)
    @patch('src.llm_handler.image_to_base64', return_value="fake_base64_string")
    def test_get_transcription_from_llm_unexpected_response_structure(self, mock_img_to_base64, mock_configure_api):
        """Test get_transcription_from_llm with unexpected API response structure."""
        import src.llm_handler as llm_handler_module_for_test
        llm_handler_module_for_test.OPENAI_API_CONFIGURED = True
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [] 
        mock_client.chat.completions.create.return_value = mock_response
        llm_handler_module_for_test.client = mock_client
        
        transcription = get_transcription_from_llm(self.test_image_path, "prompt")
        self.assertIsNone(transcription)

if __name__ == '__main__':
    unittest.main() 