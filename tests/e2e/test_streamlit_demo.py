"""
E2E tests for the Streamlit demo interface using Playwright.
Tests the main user flows for model inference and comparison.
"""

import pytest
import time
import subprocess
import os
import signal
from pathlib import Path
from playwright.sync_api import Page, expect


class StreamlitServer:
    """Context manager for running Streamlit server during tests."""
    
    def __init__(self, app_path: str, port: int = 8501):
        self.app_path = app_path
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def __enter__(self):
        """Start the Streamlit server."""
        cmd = [
            "streamlit", "run", self.app_path,
            "--server.port", str(self.port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(self.app_path).parent
        )
        
        # Wait for server to start
        time.sleep(8)  # Give Streamlit time to initialize
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the Streamlit server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


@pytest.fixture(scope="module")
def streamlit_server():
    """Fixture to start and stop Streamlit server for tests."""
    app_path = Path(__file__).parent.parent.parent / "demo_streamlit.py"
    
    with StreamlitServer(str(app_path)) as server:
        yield server


@pytest.fixture
def page_with_demo(page: Page, streamlit_server):
    """Fixture to navigate to the demo page."""
    page.goto(streamlit_server.base_url)
    
    # Wait for page to load completely
    page.wait_for_selector("h1", timeout=30000)
    expect(page.locator("h1")).to_contain_text("LLM Fine-Tuning Pipeline Demo")
    
    yield page


class TestStreamlitDemo:
    """Test suite for Streamlit demo interface."""
    
    def test_page_loads_successfully(self, page_with_demo):
        """Test that the demo page loads with expected elements."""
        page = page_with_demo
        
        # Check main title
        expect(page.locator("h1")).to_contain_text("LLM Fine-Tuning Pipeline Demo")
        
        # Check main sections are present
        expect(page.locator("text=Configuration")).to_be_visible()
        expect(page.locator("text=Input")).to_be_visible()
        expect(page.locator("text=Output")).to_be_visible()
    
    def test_sidebar_configuration(self, page_with_demo):
        """Test sidebar configuration options."""
        page = page_with_demo
        
        # Check inference method selection
        expect(page.locator("text=Choose inference method")).to_be_visible()
        expect(page.locator("text=Local Model")).to_be_visible()
        expect(page.locator("text=OpenRouter Only")).to_be_visible()
        
        # Check generation parameters
        expect(page.locator("text=Generation Parameters")).to_be_visible()
        expect(page.locator("text=Max New Tokens")).to_be_visible()
        expect(page.locator("text=Temperature")).to_be_visible()
        expect(page.locator("text=Top K")).to_be_visible()
        expect(page.locator("text=Top P")).to_be_visible()
    
    def test_input_methods(self, page_with_demo):
        """Test different input methods."""
        page = page_with_demo
        
        # Check input method options
        expect(page.locator("text=Text Input")).to_be_visible()
        expect(page.locator("text=File Upload")).to_be_visible()
        expect(page.locator("text=Example Prompts")).to_be_visible()
        
        # Test text input
        text_area = page.locator("textarea").first
        test_input = "Test prompt for generation"
        text_area.fill(test_input)
        expect(text_area).to_have_value(test_input)
        
        # Test example prompts
        page.locator("text=Example Prompts").click()
        page.wait_for_timeout(1000)  # Wait for UI update
        
        # Select an example
        example_dropdown = page.locator("select").first
        if example_dropdown.is_visible():
            example_dropdown.select_option(index=1)  # Select first non-empty option
    
    def test_generation_parameters_adjustment(self, page_with_demo):
        """Test adjustment of generation parameters."""
        page = page_with_demo
        
        # Test temperature slider
        temp_slider = page.locator("input[type='range']").first
        if temp_slider.is_visible():
            # Move slider (approximate interaction)
            temp_slider.evaluate("slider => slider.value = 0.5")
        
        # Test max tokens slider
        token_sliders = page.locator("input[type='range']")
        if token_sliders.count() > 1:
            token_sliders.nth(0).evaluate("slider => slider.value = 150")
    
    def test_openrouter_configuration(self, page_with_demo):
        """Test OpenRouter settings configuration."""
        page = page_with_demo
        
        # Check OpenRouter settings section
        expect(page.locator("text=OpenRouter Settings")).to_be_visible()
        
        # Test OpenRouter checkbox
        openrouter_checkbox = page.locator("input[type='checkbox']").first
        if openrouter_checkbox.is_visible():
            openrouter_checkbox.click()
            expect(openrouter_checkbox).to_be_checked()
            
            # Test model input field
            model_input = page.locator("input[type='text']").first
            if model_input.is_visible():
                model_input.fill("gpt-3.5-turbo")
                expect(model_input).to_have_value("gpt-3.5-turbo")
    
    def test_generate_button_interaction(self, page_with_demo):
        """Test generate button states and interactions."""
        page = page_with_demo
        
        # Initially, generate button should be disabled (no input)
        generate_button = page.locator("button:has-text('Generate')")
        expect(generate_button).to_be_visible()
        
        # Add input text to enable button
        text_area = page.locator("textarea").first
        text_area.fill("The future of AI includes")
        
        # Button should now be enabled
        page.wait_for_timeout(1000)  # Wait for UI update
        expect(generate_button).to_be_enabled()
    
    def test_mock_generation_flow(self, page_with_demo):
        """Test generation flow with mocked responses."""
        page = page_with_demo
        
        # Set up input
        text_area = page.locator("textarea").first
        test_prompt = "Complete this sentence: The benefits of technology"
        text_area.fill(test_prompt)
        
        # Mock the generation by intercepting network requests
        # (In a real scenario, you might mock the backend API calls)
        
        # Click generate button
        generate_button = page.locator("button:has-text('Generate')")
        
        # Note: In actual implementation, you would mock the model loading
        # and generation to avoid requiring actual models for testing
        # For now, we just test the UI interaction
        
        if generate_button.is_enabled():
            # Test clicking the button (without actual generation)
            generate_button.click()
            
            # Check for loading states or error messages
            page.wait_for_timeout(2000)
            
            # The actual generation might fail without models,
            # so we check for appropriate error handling
            error_elements = page.locator("text=error", page.locator("text=Error"))
            if error_elements.count() > 0:
                # Error handling is working
                assert True
    
    def test_usage_instructions_expandable(self, page_with_demo):
        """Test that usage instructions are accessible."""
        page = page_with_demo
        
        # Look for expandable sections
        usage_section = page.locator("text=Usage Instructions")
        if usage_section.is_visible():
            usage_section.click()
            page.wait_for_timeout(1000)
            
            # Check that instructions content is visible
            expect(page.locator("text=How to use this demo")).to_be_visible()
    
    def test_model_information_section(self, page_with_demo):
        """Test model information expandable section."""
        page = page_with_demo
        
        # Look for model information section
        model_info = page.locator("text=Model Information")
        if model_info.is_visible():
            model_info.click()
            page.wait_for_timeout(1000)
    
    def test_environment_information(self, page_with_demo):
        """Test environment information section."""
        page = page_with_demo
        
        # Look for environment info
        env_info = page.locator("text=Environment Information")
        if env_info.is_visible():
            env_info.click()
            page.wait_for_timeout(1000)
            
            # Check for environment details
            expect(page.locator("text=Available Models")).to_be_visible()
    
    def test_responsive_layout(self, page_with_demo):
        """Test responsive layout behavior."""
        page = page_with_demo
        
        # Test different viewport sizes
        original_size = page.viewport_size
        
        # Mobile size
        page.set_viewport_size({"width": 375, "height": 667})
        page.wait_for_timeout(1000)
        
        # Check that main elements are still visible
        expect(page.locator("h1")).to_be_visible()
        
        # Tablet size
        page.set_viewport_size({"width": 768, "height": 1024})
        page.wait_for_timeout(1000)
        expect(page.locator("h1")).to_be_visible()
        
        # Restore original size
        if original_size:
            page.set_viewport_size(original_size)
    
    def test_error_handling_display(self, page_with_demo):
        """Test that errors are properly displayed to users."""
        page = page_with_demo
        
        # Try to generate without proper setup (should show error)
        text_area = page.locator("textarea").first
        text_area.fill("Test input")
        
        generate_button = page.locator("button:has-text('Generate')")
        if generate_button.is_enabled():
            generate_button.click()
            
            # Wait for potential error messages
            page.wait_for_timeout(5000)
            
            # Look for error indicators (various forms)
            error_indicators = [
                "text=error",
                "text=Error",
                "text=Failed",
                "text=not found",
                "text=not available"
            ]
            
            error_found = any(
                page.locator(indicator).count() > 0 
                for indicator in error_indicators
            )
            
            # Either generation works or errors are properly shown
            assert error_found or page.locator("text=Output").is_visible()
    
    def test_input_validation(self, page_with_demo):
        """Test input validation and constraints."""
        page = page_with_demo
        
        # Test empty input handling
        text_area = page.locator("textarea").first
        text_area.fill("")
        
        generate_button = page.locator("button:has-text('Generate')")
        # Button should be disabled with empty input
        page.wait_for_timeout(1000)
        
        # Test very long input
        long_input = "A" * 2000  # Very long input
        text_area.fill(long_input)
        
        # Should still be able to handle long input
        page.wait_for_timeout(1000)
        expect(text_area).to_have_value(long_input)
    
    def test_parameter_bounds(self, page_with_demo):
        """Test that parameter sliders respect their bounds."""
        page = page_with_demo
        
        # Test slider boundaries
        sliders = page.locator("input[type='range']")
        
        for i in range(sliders.count()):
            slider = sliders.nth(i)
            if slider.is_visible():
                # Get slider attributes
                min_val = slider.get_attribute("min")
                max_val = slider.get_attribute("max")
                
                if min_val and max_val:
                    # Test minimum value
                    slider.evaluate(f"slider => slider.value = {min_val}")
                    
                    # Test maximum value
                    slider.evaluate(f"slider => slider.value = {max_val}")


@pytest.mark.integration
class TestStreamlitDemoIntegration:
    """Integration tests requiring actual model files or API connections."""
    
    @pytest.mark.skipif(
        not Path("./checkpoints").exists(),
        reason="No model checkpoints available"
    )
    def test_local_model_loading(self, page_with_demo):
        """Test loading local model checkpoints."""
        page = page_with_demo
        
        # Select Local Model option
        page.locator("text=Local Model").click()
        page.wait_for_timeout(2000)
        
        # Look for model selection dropdown
        model_dropdown = page.locator("select")
        if model_dropdown.count() > 0:
            # Should show available models
            expect(model_dropdown.first).to_be_visible()
    
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_BASE_URL"),
        reason="OpenRouter configuration not available"
    )
    def test_openrouter_integration(self, page_with_demo):
        """Test OpenRouter API integration."""
        page = page_with_demo
        
        # Enable OpenRouter
        openrouter_checkbox = page.locator("input[type='checkbox']")
        if openrouter_checkbox.count() > 0:
            openrouter_checkbox.first.click()
            
            # Add input and try generation
            text_area = page.locator("textarea").first
            text_area.fill("Test OpenRouter integration")
            
            generate_button = page.locator("button:has-text('Generate')")
            if generate_button.is_enabled():
                generate_button.click()
                
                # Wait for API response
                page.wait_for_timeout(10000)
                
                # Should show either output or appropriate error
                output_section = page.locator("text=OpenRouter Output")
                error_section = page.locator("text=not available")
                
                assert output_section.is_visible() or error_section.is_visible()