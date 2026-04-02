import pytest
from unittest.mock import Mock, patch
from superagi.llms.llm_model_factory import get_model
from superagi.llms.llama3 import Llama3

@patch('superagi.llms.llm_model_factory.connect_db')
def test_get_model_with_llama3(mock_connect_db):
    # Arrange
    mock_engine = Mock()
    mock_connect_db.return_value = mock_engine
    mock_session = Mock()
    mock_engine.session.return_value = mock_session
    mock_model_instance = Mock()
    mock_model_instance.model_name = "llama3-70b-8192"
    mock_model_instance.model_provider_id = 1
    mock_session.query.return_value.filter.return_value.first.return_value = mock_model_instance
    mock_provider = Mock()
    mock_provider.provider = "Llama3"
    mock_session.query.return_value.filter.return_value.first.return_value = mock_provider

    # Act
    model = get_model(organisation_id=1, api_key="fake_key", model="llama3-70b-8192")

    # Assert
    assert isinstance(model, Llama3)
