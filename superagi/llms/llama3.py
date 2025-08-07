"""Llama3 LLM."""
from llama_api_client import LlamaAPIClient, APIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm

MAX_RETRY_ATTEMPTS = 5
MIN_WAIT = 30 # Seconds
MAX_WAIT = 300 # Seconds

def custom_retry_error_callback(retry_state):
    """
    Custom retry error callback.

    Args:
        retry_state (tenacity.RetryCallState): The retry state.

    Returns:
        dict: The error message.
    """
    logger.info("Llama3 Exception:", retry_state.outcome.exception())
    return {"error": "ERROR_LLAMA3", "message": "Llama3 exception: "+str(retry_state.outcome.exception())}


class Llama3(BaseLlm):
    """
    Llama3 LLM.

    Args:
        api_key (str): The Llama API key.
        model (str): The model.
        temperature (float): The temperature.
        max_tokens (int): The maximum number of tokens.
        top_p (float): The top p.
        frequency_penalty (float): The frequency penalty.
        presence_penalty (float): The presence penalty.
        number_of_results (int): The number of results.
    """
    def __init__(self, api_key, model="llama3-70b-8192", temperature=0.6,
                 max_tokens=get_config("MAX_MODEL_TOKEN_LIMIT"), top_p=1,
                 frequency_penalty=0, presence_penalty=0, number_of_results=1):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.number_of_results = number_of_results
        self.api_key = api_key
        self.client = LlamaAPIClient(api_key=self.api_key)

    def get_source(self):
        return "llama3"

    def get_api_key(self):
        """
        Returns:
            str: The API key.
        """
        return self.api_key

    def get_model(self):
        """
        Returns:
            str: The model.
        """
        return self.model

    @retry(
        retry=retry_if_exception_type(APIError),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),  # Maximum number of retry attempts
        wait=wait_random_exponential(min=MIN_WAIT, max=MAX_WAIT),
        before_sleep=lambda retry_state: logger.info(
            f"{retry_state.outcome.exception()} (attempt {retry_state.attempt_number})"),
        retry_error_callback=custom_retry_error_callback
    )
    def chat_completion(self, messages, max_tokens=get_config("MAX_MODEL_TOKEN_LIMIT")):
        """
        Call the Llama3 chat completion API.

        Args:
            messages (list): The messages.
            max_tokens (int): The maximum number of tokens.

        Returns:
            dict: The response.
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=self.top_p,
                # The Llama API does not support frequency_penalty and presence_penalty
            )
            content = response.choices[0].message["content"]
            return {"response": response, "content": content}
        except APIError as api_error:
            logger.info("Llama3 APIError:", api_error)
            raise api_error
        except Exception as exception:
            logger.info("Llama3 Exception:", exception)
            return {"error": "ERROR_LLAMA3", "message": "Llama3 exception: "+str(exception)}

    def verify_access_key(self):
        """
        Verify the access key is valid.

        Returns:
            bool: True if the access key is valid, False otherwise.
        """
        # The Llama API does not have a dedicated endpoint for listing models.
        # We can try to make a simple API call to check if the key is valid.
        try:
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "hello"}],
                model=self.model,
                max_tokens=1
            )
            return True
        except Exception as exception:
            logger.info("Llama3 Exception:", exception)
            return False

    def get_models(self):
        """
        Get the models.

        Returns:
            list: The models.
        """
        # The Llama API does not have a dedicated endpoint for listing models.
        # We will return a hardcoded list of supported models.
        return ["llama3-70b-8192", "llama3-8b-8192"]
