import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load env and initialize client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not set.")
    exit(1)
client = OpenAI(api_key=api_key)

# Check available models (for debugging; you can remove in production)
try:
    resp = client.models.list()
    available = [m.id for m in resp.data]
    print("Available models:", available)
except Exception as e:
    print("Could not list models:", e)

# Choose a model your account has access to
MODEL_NAME = "gpt-3.5-turbo"  # replace if you do have GPT-4 access, e.g. "gpt-4" or "gpt-4o"

# Load dataset
df = pd.read_csv("pizza_dataset.csv")
print("CSV columns:", df.columns.tolist())
for col in ("category", "name"):
    if col not in df.columns:
        print(f"Error: column '{col}' missing in CSV")
        exit(1)

named_pizzas = df[df["category"] == "named_pizza"]
sizes = df[df["category"] == "size"]["name"].tolist()
crusts = df[df["category"] == "crust"]["name"].tolist()
toppings = df[df["category"] == "topping"]["name"].tolist()
sauces = df[df["category"] == "sauce"]["name"].tolist()
named_pizza_names = named_pizzas["name"].str.lower().tolist()

# --- Order Function ---
def order_pizza(size, crust, toppings, sauces):
    base_price = {"small": 5, "medium": 7, "large": 9, "extra large": 11}
    price = base_price.get(size.lower(), 7)
    price += 0.75 * len(toppings) + 0.5 * len(sauces)
    return {
        "status": "success",
        "order": {
            "size": size,
            "crust": crust,
            "toppings": toppings,
            "sauces": sauces,
            "price": round(price, 2)
        },
        "eta": "20 minutes"
    }

order_tool = {
    "name": "order_pizza",
    "description": "Place a pizza order with size, crust, toppings, and sauces",
    "parameters": {
        "type": "object",
        "required": ["size", "crust", "toppings", "sauces"],
        "properties": {
            "size": {"type": "string"},
            "crust": {"type": "string"},
            "toppings": {"type": "array", "items": {"type": "string"}},
            "sauces": {"type": "array", "items": {"type": "string"}}
        }
    }
}

# --- Name Extraction ---
def extract_name(user_text):
    system_msg = "Extract only the person's name from the message. Reply with only the name."
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_text}
            ],
            temperature=0.0,
        )
    except Exception as e:
        raise RuntimeError(f"Name-extraction failed: {e}")
    return resp.choices[0].message.content.strip()

# --- Main Function ---
def main():
    print("👋 Welcome to Pizza Bot!")
    raw_name_input = input("Tell me your name: ").strip()
    if not raw_name_input:
        print("No name entered. Exiting.")
        return

    try:
        name = extract_name(raw_name_input)
    except Exception as e:
        print(f"❌ Error extracting name: {e}")
        # Fallback: use raw input as name
        name = raw_name_input
        print(f"Proceeding with name: {name}")

    if not name:
        name = raw_name_input
        print(f"Using raw input as name: {name}")

    print(f"Hi {name}! Let's start your order.")

    messages = [
        {"role": "system", "content": "You are a helpful pizza-ordering assistant. Ask the user questions and place their order by calling 'order_pizza' function when ready."}
    ]

    orders = []
    total = 0.0

    while True:
        user_msg = input(f"\n{name}: ").strip()
        if not user_msg:
            continue
        messages.append({"role": "user", "content": user_msg})

        while True:
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    functions=[order_tool],
                    function_call="auto",
                )
            except Exception as e:
                print(f"❌ Error during chat completion: {e}")
                return

            choice = resp.choices[0]
            msg = choice.message

            if msg.content:
                print(f"🤖 AI: {msg.content}")
                messages.append({"role": "assistant", "content": msg.content})

            if msg.function_call:
                try:
                    args = json.loads(msg.function_call.arguments)
                except json.JSONDecodeError:
                    print("❌ Failed to parse function arguments")
                    break
                result = order_pizza(**args)
                print(f"\n📦 Order placed! ETA: {result['eta']}")
                print(f"💰 Price: ${result['order']['price']:.2f}")
                orders.append(result["order"])
                total += result["order"]["price"]
                messages.append({
                    "role": "function",
                    "name": "order_pizza",
                    "content": json.dumps(result)
                })
                break

            user_msg = input(f"\n{name}: ").strip()
            if not user_msg:
                continue
            messages.append({"role": "user", "content": user_msg})

        cont = input("\nWould you like to order another pizza? (yes/no): ").strip().lower()
        if cont != "yes":
            break

    print("\n📋 Final Order Summary:")
    for i, o in enumerate(orders, 1):
        print(f"Pizza {i}: {o['size']} {o['crust']} crust")
        print(f"  Toppings: {', '.join(o['toppings'])}")
        print(f"  Sauces: {', '.join(o['sauces'])}")
        print(f"  Price: ${o['price']:.2f}\n")

    print(f"🧾 Total Bill: ${total:.2f}")
    print("🔔 Your pizzas will be ready soon. Thank you!")

if __name__ == "__main__":
    main()
