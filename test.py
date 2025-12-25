# list_models.py
import os
from dotenv import load_dotenv

load_dotenv()

# Use ADC or API key depending on your setup.
# If you rely on API key only, make sure your client library supports passing it;
# otherwise set GOOGLE_APPLICATION_CREDENTIALS to service-account JSON before running.

try:
    # import the client class used in the stack trace
    from google.ai.generativelanguage_v1beta.services.generative_service.client import GenerativeServiceClient

    client = GenerativeServiceClient()
    print("Listing models available to this credential/project:\n")
    for model in client.list_models():
        # print name; model may have more metadata you can inspect
        print(model.name)
except Exception as e:
    print("Error while listing models:")
    import traceback
    traceback.print_exc()
