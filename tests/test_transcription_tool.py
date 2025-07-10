import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import argparse
import sys
import logging
import os 

from src import transcription_tool 
from src.transcription_tool import main, create_parser, get_prompt_content, process_single_image, process_directory

# Suppress most logging output during tests
logging.disable(logging.CRITICAL)

class TestTranscriptionTool(unittest.TestCase):

    def setUp(self):
        """Setup common to tests. Reset modules if necessary."""
        # Reset any global state if transcription_tool module modifies it globally upon import
        # For example, if OPENAI_API_CONFIGURED was set at import time in transcription_tool directly
        # (though it's imported from llm_handler)
        pass

    @patch('src.transcription_tool.argparse.ArgumentParser')
    def test_create_parser(self, mock_argparse):
        """Test the create_parser function for correct argument setup."""
        mock_parser_instance = MagicMock()
        mock_add_mutually_exclusive_group = MagicMock()
        mock_parser_instance.add_mutually_exclusive_group.return_value = mock_add_mutually_exclusive_group
        mock_argparse.return_value = mock_parser_instance

        parser = transcription_tool.create_parser() # Call the function in the module

        self.assertEqual(parser, mock_parser_instance)
        mock_parser_instance.add_mutually_exclusive_group.assert_called_once_with(required=True)
        
        expected_calls_group = [
            call("--image_file", type=Path, help="Percorso del singolo file immagine JPG da processare."),
            call("--image_dir", type=Path, help="Percorso della directory contenente i file immagine JPG da processare.")
        ]
        mock_add_mutually_exclusive_group.add_argument.assert_has_calls(expected_calls_group, any_order=False)

        expected_calls_parser = [
            call("--prompt_file", type=Path, default=None, help=f"Percorso del file contenente il prompt per la trascrizione. Default: ./{transcription_tool.DEFAULT_PROMPT_FILENAME}"),
            call("--output_dir", type=Path, default=None, help="Directory dove salvare i file di trascrizione. Default: stessa directory dell'immagine/immagini o directory di input per la modalità batch.")
        ]
        mock_parser_instance.add_argument.assert_has_calls(expected_calls_parser, any_order=False)
        self.assertEqual(mock_parser_instance.add_argument.call_count, 2) # only the non-group ones

    @patch('src.transcription_tool.load_prompt_file')
    @patch('src.transcription_tool.DEFAULT_PROMPT_FILE_PATH', new_callable=MagicMock)
    def test_get_prompt_content_uses_default(self, mock_default_path, mock_load_prompt):
        """Test get_prompt_content uses default path when no argument is provided."""
        mock_default_path.exists.return_value = True # Assume default prompt exists
        mock_load_prompt.return_value = "Default prompt content"
        
        content = get_prompt_content(None)
        
        self.assertEqual(content, "Default prompt content")
        mock_load_prompt.assert_called_once_with(mock_default_path)

    @patch('src.transcription_tool.load_prompt_file')
    def test_get_prompt_content_uses_specified_path(self, mock_load_prompt):
        """Test get_prompt_content uses specified path when argument is provided."""
        specified_path = Path("custom/prompt.txt")
        mock_load_prompt.return_value = "Custom prompt content"
        
        content = get_prompt_content(specified_path)
        
        self.assertEqual(content, "Custom prompt content")
        mock_load_prompt.assert_called_once_with(specified_path)

    @patch('src.transcription_tool.load_prompt_file', return_value=None)
    @patch('src.transcription_tool.DEFAULT_PROMPT_FILE_PATH', new_callable=MagicMock)
    def test_get_prompt_content_fails_to_load(self, mock_default_path, mock_load_prompt):
        """Test get_prompt_content returns None if loading fails."""
        mock_default_path.exists.return_value = False # Simulate it not existing too
        self.assertIsNone(get_prompt_content(None))
        mock_load_prompt.assert_called_once_with(mock_default_path)

    # --- Tests for main() --- 
    # These are more integration-like for the main function's orchestration logic

    @patch('src.transcription_tool.sys.exit')
    @patch('src.transcription_tool.argparse.ArgumentParser')
    @patch('src.transcription_tool.get_prompt_content')
    @patch('src.transcription_tool.process_single_image')
    @patch('src.transcription_tool.process_directory')
    @patch('src.transcription_tool.OPENAI_API_CONFIGURED', True) # Assume API configured for these main tests
    @patch('pathlib.Path.mkdir') # Mock mkdir for output_dir
    def test_main_single_file_success(self, mock_mkdir, mock_proc_dir, mock_proc_single, mock_get_prompt, mock_arg_parser, mock_sys_exit):
        """Test main function orchestrates single file processing successfully."""
        mock_args = MagicMock()
        mock_args.image_file = Path("test.jpg")
        mock_args.image_dir = None
        mock_args.prompt_file = None
        mock_args.output_dir = Path("output")
        mock_arg_parser.return_value.parse_args.return_value = mock_args
        
        mock_get_prompt.return_value = "Mocked prompt text"
        mock_proc_single.return_value = True # Successful processing

        # Mock is_file for the input image
        with patch.object(Path, 'is_file', return_value=True) as mock_is_file_method:
            main()
            mock_is_file_method.assert_any_call() # Called on args.image_file

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True) # For output_dir
        mock_get_prompt.assert_called_once_with(None)
        mock_proc_single.assert_called_once_with(Path("test.jpg"), "Mocked prompt text", Path("output"))
        mock_proc_dir.assert_not_called()
        mock_sys_exit.assert_called_once_with(0)

    @patch('src.transcription_tool.sys.exit')
    @patch('src.transcription_tool.argparse.ArgumentParser')
    @patch('src.transcription_tool.get_prompt_content')
    @patch('src.transcription_tool.process_single_image')
    @patch('src.transcription_tool.process_directory')
    @patch('src.transcription_tool.OPENAI_API_CONFIGURED', True)
    @patch('pathlib.Path.mkdir')
    def test_main_directory_success(self, mock_mkdir, mock_proc_dir, mock_proc_single, mock_get_prompt, mock_arg_parser, mock_sys_exit):
        """Test main function orchestrates directory processing successfully."""
        mock_args = MagicMock()
        mock_args.image_file = None
        mock_args.image_dir = Path("img_dir")
        mock_args.prompt_file = Path("custom_prompt.txt")
        mock_args.output_dir = None # Test default output dir for directory mode
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        mock_get_prompt.return_value = "Custom prompt"
        mock_proc_dir.return_value = True # Successful directory processing

        # Mock is_dir for the input directory
        with patch.object(Path, 'is_dir', return_value=True) as mock_is_dir_method:
            main()
            mock_is_dir_method.assert_any_call() # Called on args.image_dir
        
        mock_mkdir.assert_not_called() # output_dir is None, mkdir called inside process_directory if needed
        mock_get_prompt.assert_called_once_with(Path("custom_prompt.txt"))
        mock_proc_dir.assert_called_once_with(Path("img_dir"), "Custom prompt", None)
        mock_proc_single.assert_not_called()
        mock_sys_exit.assert_called_once_with(0)

    @patch('src.transcription_tool.argparse.ArgumentParser')
    @patch('src.transcription_tool.OPENAI_API_CONFIGURED', False) 
    def test_main_api_not_configured_exits(self, mock_arg_parser):
        """Test main exits early if OpenAI API is not configured."""
        with patch('src.transcription_tool.sys.exit', side_effect=SystemExit(1)) as mock_sys_exit:
            # Setup parser to avoid errors, though it shouldn't get far
            mock_args = MagicMock()
            mock_args.image_file = Path("test.jpg") 
            mock_args.image_dir = None
            mock_arg_parser.return_value.parse_args.return_value = mock_args

            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)
            mock_sys_exit.assert_called_once_with(1)

    @patch('src.transcription_tool.argparse.ArgumentParser')
    @patch('src.transcription_tool.get_prompt_content', return_value=None) # Prompt loading fails
    @patch('src.transcription_tool.OPENAI_API_CONFIGURED', True)
    @patch('pathlib.Path.mkdir')
    def test_main_prompt_load_fails_exits(self, mock_mkdir, mock_get_prompt, mock_arg_parser):
        """Test main exits if prompt content cannot be loaded."""
        with patch('src.transcription_tool.sys.exit', side_effect=SystemExit(1)) as mock_sys_exit:
            mock_args = MagicMock()
            mock_args.image_file = Path("test.jpg")
            mock_args.image_dir = None
            mock_args.prompt_file = Path("bad_prompt.txt")
            mock_args.output_dir = Path("output")
            mock_arg_parser.return_value.parse_args.return_value = mock_args
            
            # Mock is_file for the input image to allow reaching prompt loading
            with patch.object(Path, 'is_file', return_value=True):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_get_prompt.assert_called_once_with(Path("bad_prompt.txt"))
            mock_sys_exit.assert_called_once_with(1)

    @patch('src.transcription_tool.sys.exit')
    @patch('src.transcription_tool.argparse.ArgumentParser')
    @patch('src.transcription_tool.get_prompt_content')
    @patch('src.transcription_tool.process_single_image', return_value=False) # Single image processing fails
    @patch('src.transcription_tool.OPENAI_API_CONFIGURED', True)
    @patch('pathlib.Path.mkdir')
    def test_main_single_file_processing_fails_exits(self, mock_mkdir, mock_proc_single, mock_get_prompt, mock_arg_parser, mock_sys_exit):
        """Test main exits with error if single file processing fails."""
        mock_args = MagicMock()
        mock_args.image_file = Path("test.jpg")
        mock_args.image_dir = None
        mock_args.prompt_file = None
        mock_args.output_dir = None
        mock_arg_parser.return_value.parse_args.return_value = mock_args
        
        mock_get_prompt.return_value = "Mocked prompt"
        
        with patch.object(Path, 'is_file', return_value=True):
            main()
        
        mock_proc_single.assert_called_once_with(Path("test.jpg"), "Mocked prompt", None)
        mock_sys_exit.assert_called_once_with(1)

    @patch('src.transcription_tool.argparse.ArgumentParser')
    def test_main_input_file_not_exists(self, mock_arg_parser):
        """Test main exits if specified image_file does not exist."""
        with patch('src.transcription_tool.sys.exit', side_effect=SystemExit(1)) as mock_sys_exit:
            mock_args = MagicMock()
            mock_args.image_file = Path("nonexistent.jpg")
            mock_args.image_dir = None
            mock_args.prompt_file = None
            mock_args.output_dir = None
            mock_arg_parser.return_value.parse_args.return_value = mock_args

            # Patch OPENAI_API_CONFIGURED and get_prompt_content to allow execution up to file check
            with patch('src.transcription_tool.OPENAI_API_CONFIGURED', True), \
                 patch('src.transcription_tool.get_prompt_content', return_value="prompt"), \
                 patch.object(Path, 'is_file', return_value=False) as mock_is_file:
                
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
                mock_is_file.assert_called_once_with() # Called on nonexistent.jpg
            mock_sys_exit.assert_called_once_with(1)

    @patch('src.transcription_tool.argparse.ArgumentParser')
    @patch('pathlib.Path.mkdir', side_effect=PermissionError("Cannot create dir"))
    def test_main_output_dir_creation_permission_error(self, mock_mkdir, mock_arg_parser):
        """Test main exits if output_dir creation fails due to permissions."""
        with patch('src.transcription_tool.sys.exit', side_effect=SystemExit(1)) as mock_sys_exit:
            mock_args = MagicMock()
            mock_args.image_file = Path("test.jpg") 
            mock_args.image_dir = None
            mock_args.prompt_file = None
            mock_args.output_dir = Path("restricted_output")
            mock_arg_parser.return_value.parse_args.return_value = mock_args

            with patch('src.transcription_tool.OPENAI_API_CONFIGURED', True):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)
            
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_sys_exit.assert_called_once_with(1)

if __name__ == '__main__':
    unittest.main() 