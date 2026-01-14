import dotenv
import sam3

dotenv.load_dotenv('/workspace/sam3/.env')
import os

hf_token = os.getenv('TOKENHF')
print("Hugging Face Token:", hf_token)

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
print("SAM3 Root Directory:", sam3_root)