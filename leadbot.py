import csv
from datetime import datetime
from transformers import pipeline
import gradio as gr
import pandas as pd
import os
import matplotlib.pyplot as plt 
from dotenv import load_dotenv

load_dotenv()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# NEW IMPORTS FOR LOCAL RAG
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Lead qualification labels 
LABELS = {
    "product_interest": ["electronics", "fashion", "home appliances", "beauty", "books", "sports", "other"],
    "budget": ["under 500", "500 to 1000", "above 1000"],
    "urgency": ["today", "within a week", "this month", "just Browse"],
    "intent": ["ready to buy", "comparing", "window shopping", "just Browse"]
}

# Lead qualification questions
QUESTIONS = {
    "name": "What's your name?",
    "email": "What's your email?",
    "product_interest": "What kind of product are you interested in? (electronics, fashion, home applicances, beauty, books, sports)",
    "budget": "What’s your estimated budget?",
    "urgency": "When are you planning to buy?",
    "intent": "Are you ready to make a purchase or just exploring?"
}

QUALIFICATION_QUESTION_KEYS = ["name", "email", "product_interest", "budget", "urgency", "intent"]


# Lead scoring logic
def score_lead(answers):
    score = 0
    if answers["budget"] == "above 1000":
        score += 30
    elif answers["budget"] == "500 to 1000":
        score += 20
    else:
        score += 10

    if answers["urgency"] in ["today", "within a week"]:
        score += 30
    elif answers["urgency"] == "this month":
        score += 15

    if answers["intent"] == "ready to buy":
        score += 30
    elif answers["intent"] == "comparing":
        score += 20

    return score

# Lead segmentation logic
def segment_lead(score):
    if score >= 80:
        return "Hot Lead"
    elif score >= 50:
        return "Warm Lead"
    else:
        return "Cold Lead"

# Lead qualifier
def qualify_and_score_lead_internal(temp_answers):
    name = temp_answers.get("name", "N/A")
    email = temp_answers.get("email", "N/A")

    parsed_answers = {}
    interpretation_messages = []

    for key in ["product_interest", "budget", "urgency", "intent"]:
        input_value = temp_answers.get(key, "")
        if key in LABELS:
            if input_value:
                result = classifier(input_value, LABELS[key])
                parsed_answers[key] = result["labels"][0]
                interpretation_messages.append(f"Interpreted {key.replace('_', ' ').title()} as: {parsed_answers[key]}")
            else:
                parsed_answers[key] = "N/A" # Handle cases where input might be empty
                interpretation_messages.append(f"No input for {key.replace('_', ' ').title()}.")
        else:
            parsed_answers[key] = input_value

    score = score_lead(parsed_answers)
    segment = segment_lead(score)

    filename = "shopping_leads.csv"
    fieldnames = ["name", "email", "product_interest", "budget", "urgency", "intent", "score", "segment", "timestamp"]

    try:
        with open(filename, 'x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass 

    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        row = {
            "name": name,
            "email": email,
            "product_interest": parsed_answers.get("product_interest", "N/A"),
            "budget": parsed_answers.get("budget", "N/A"),
            "urgency": parsed_answers.get("urgency", "N/A"),
            "intent": parsed_answers.get("intent", "N/A"),
            "score": score,
            "segment": segment,
            "timestamp": datetime.utcnow().isoformat()
        }
        writer.writerow(row)
    
    output_message = "<br>".join(interpretation_messages)
    output_message += f"<br><br>Lead Score: {score}/100<br>Segment: {segment}"
    output_message += "<br>Lead saved successfully!"

    # --- Add Product Recommendation based on qualified lead data ---
    qualified_product_interest = parsed_answers.get("product_interest", "N/A")
    qualified_budget = parsed_answers.get("budget", "N/A")
    qualified_urgency = parsed_answers.get("urgency", "N/A") 

    if qualified_product_interest != "N/A" and qualified_product_interest != "other":
        product_recommendation_text = recommend_products_gradio(
            qualified_product_interest,
            qualified_budget,
            qualified_urgency
        )
        output_message += f"<br><br>Here's a product recommendation based on your details:<br>{product_recommendation_text}"
    else:
        output_message += "<br><br>We couldn't provide a specific product recommendation as your product interest was not defined or categorized."
    
    return output_message


# FAQ knowledge base
FAQS = {
    "shipping": {
        "keywords": ["shipping", "delivery", "arrival", "ship", "lost", "missing", "where is my order", "not arrived"],
        "answer": "We offer free shipping on orders over ₹499. Delivery usually takes 3–5 business days. If your order is lost or hasn't arrived, please contact our support team immediately at help@shopsmart.com or call 1800-123-456 for assistance."
    },
    "returns": {
        "keywords": ["return", "refund", "exchange", "cancel order"],
        "answer": "You can return any item within 10 days. Refunds are processed within 3–5 business days."
    },
    "payment": {
        "keywords": ["payment", "methods", "pay", "upi", "credit card"],
        "answer": "We accept UPI, credit/debit cards, and net banking."
    },
    "support": {
        "keywords": ["support", "help", "contact", "customer service"],
        "answer": "You can reach us at help@shopsmart.com or call 1800-123-456."
    },
    "offers": {
        "keywords": ["discount", "offers", "coupon", "sale"],
        "answer": "We have seasonal offers! Check our homepage or subscribe for exclusive deals."
    }
}


# Documents from your FAQS 
rag_documents = []
for topic, data in FAQS.items():
    # Combine keywords and answer for richer embedding context
    combined_content = f"Topic: {topic.replace('_', ' ').title()}. Keywords: {', '.join(data['keywords'])}. Answer: {data['answer']}"
    rag_documents.append(Document(page_content=combined_content, metadata={"topic": topic, "answer": data["answer"]}))

#Embedding Model and Vector Store (ChromaDB)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model_kwargs = {'device': 'cpu'}
embedding_encode_kwargs = {'normalize_embeddings': False} # Normalize embeddings for better similarity search

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

# create a local folder for your vector store
vectorstore = Chroma.from_documents(documents=rag_documents, embedding=embeddings)


# 3. Initialize LLM
llm = ChatOllama(model="llama3.2") 


#Prompt Template for the RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer service assistant for ShopSmart. Answer the user's question ONLY based on the following context. If the context does not contain enough information to answer the question, politely state that you cannot answer from the provided information and suggest contacting customer support. Do not make up information."),
    ("human", "Context: {context}\n\nQuestion: {input}")
])

#document chain (combines context with prompt and LLM) 
document_chain = create_stuff_documents_chain(llm, prompt)

# retrieval chain (retriever + document chain)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# --- Chat function for FAQ interface using RAG ---
def faq_chat_interface(message, history):
    if not message or message.strip() == "":
        return "Please type your question. I'm here to help with common queries about shipping, returns, payments, and more!"

    message_lower = message.lower()

    if message_lower in ["exit", "quit", "back to menu", "main menu"]:
        return "Okay, returning to the main menu. You can click the 'Back to Main Menu' button below."
    
    try:
        print(f"\nUser query: '{message}'") # Diagnostic print
        
        # Invoke the RAG chain
        response = retrieval_chain.invoke({"input": message})
        
        print(f"RAG Chain Response: {response}") # Diagnostic print: See full response
        
        retrieved_context = response.get('context', [])
        llm_answer = response.get('answer', '')

        print(f"Retrieved Context: {retrieved_context}") # Diagnostic print: What documents were found?
        print(f"LLM's Generated Answer: '{llm_answer}'") # Diagnostic print: What did the LLM produce?

        # Check if a relevant document was found and if the LLM provided a useful answer
        # You might need to adjust the condition for what constitutes a "useful answer"
        if retrieved_context and llm_answer and "contact customer support" not in llm_answer.lower():
            # For a deeper check, ensure the retrieved document's content isn't trivial
            if any(doc.page_content.strip() != "" for doc in retrieved_context):
                 return llm_answer
            else:
                 print("Retrieved context was empty or only whitespace after check, despite being non-empty list.")
                 return "I apologize, I'm having a little trouble finding a direct answer to that specific question from my knowledge base. Could you please rephrase it, or for more complex issues, please contact our support team at help@shopsmart.com or call 1800-123-456."
        else:
            print("Fallback triggered: No context, empty answer, or LLM suggested support within its answer.")
            return "I apologize, I'm having a little trouble finding a direct answer to that specific question from my knowledge base. Could you please rephrase it, or for more complex issues, please contact our support team at help@shopsmart.com or call 1800-123-456."
            
    except Exception as e:
        print(f"Error during RAG processing: {e}")
        # General fallback if RAG fails for any reason (e.g., API error, model error)
        return "I apologize, I'm currently experiencing technical difficulties. Please try again in a moment, or contact our support team at help@shopsmart.com."


# Product recommendation system 
def recommend_products_gradio(category, budget_preference, urgency_preference):
    product_catalog = {
        "electronics": {
            "Mobile Phones": [
                {"name": "iPhone 15 Pro Max", "price": 139900, "features": ["256GB", "A17 Pro", "Pro Camera System"]},
                {"name": "Samsung Galaxy S24 Ultra", "price": 129999, "features": ["5G", "S Pen", "200MP Camera"]},
                {"name": "OnePlus 12", "price": 64999, "features": ["Snapdragon 8 Gen 3", "5400mAh", "Hasselblad Camera"]},
                {"name": "Redmi Note 13 Pro", "price": 18999, "features": ["5G", "200MP OIS Camera", "120Hz AMOLED"]},
                {"name": "Nokia C22", "price": 7999, "features": ["6.5'' HD+", "5000mAh", "Durable Design"]}
            ],
            "Laptops": [
                {"name": "MacBook Air M3", "price": 114900, "features": ["M3 Chip", "Liquid Retina", "Fanless Design"]},
                {"name": "HP Pavilion Aero 13", "price": 72999, "features": ["Ryzen 7", "Lightweight", "16GB RAM"]},
                {"name": "Acer Aspire 3", "price": 38999, "features": ["Core i3", "8GB RAM", "256GB SSD"]}
            ],
            "Accessories": [
                {"name": "Sony WH-1000XM5", "price": 29999, "features": ["Noise Cancellation", "30hr Battery"]},
                {"name": "Amazfit GTS 4", "price": 14999, "features": ["AMOLED", "GPS", "Bluetooth Calling"]},
                {"name": "JBL Flip 6", "price": 9999, "features": ["Waterproof", "Portable", "Powerful Sound"]},
                {"name": "Mi Power Bank 3i", "price": 899, "features": ["10000mAh", "Fast Charging", "Dual Input"]}
            ]
        },
        "fashion": {
            "Men": [
                {"name": "Designer Blazer", "price": 4999, "features": ["Premium Fabric", "Slim Fit", "All Seasons"]},
                {"name": "Leather Wallet", "price": 1499, "features": ["Genuine Leather", "Multiple Slots", "RFID Protection"]},
                {"name": "Cotton T-Shirt", "price": 599, "features": ["100% Cotton", "Available in 5 colors"]},
                {"name": "Slim Fit Jeans", "price": 1299, "features": ["Stretchable", "Dark Wash"]},
                {"name": "Sports Socks (3-pack)", "price": 299, "features": ["Breathable", "Cushioned", "Ankle Length"]}
            ],
            "Women": [
                {"name": "Evening Gown", "price": 7999, "features": ["Silk Blend", "Embroidered", "Elegant Fit"]},
                {"name": "Smart Watch for Women", "price": 2499, "features": ["Heart Rate", "Sleep Tracking", "Stylish Design"]},
                {"name": "Floral Midi Dress", "price": 1499, "features": ["Rayon", "Ideal for Summer"]},
                {"name": "Tote Bag", "price": 899, "features": ["PU Leather", "Spacious & Stylish"]},
                {"name": "Silver Earrings", "price": 499, "features": ["Sterling Silver", "Hypoallergenic", "Stud Design"]}
            ]
        },
        "books": {
            "Self-help": [
                {"name": "Atomic Habits", "price": 449, "features": ["James Clear", "Change your habits"]},
                {"name": "The 7 Habits of Highly Effective People", "price": 650, "features": ["Stephen Covey", "Classic Self-Help", "Personal Growth"]},
                {"name": "Deep Work", "price": 399, "features": ["Cal Newport", "Focus for success"]},
                {"name": "Think and Grow Rich", "price": 720, "features": ["Napoleon Hill", "Wealth Building", "Timeless Principles"]}
            ],
            "Fiction": [
                {"name": "Project Hail Mary", "price": 580, "features": ["Andy Weir", "Sci-Fi Adventure", "Humorous"]},
                {"name": "The Midnight Library", "price": 499, "features": ["Matt Haig", "Existential Fiction", "Bestseller"]},
                {"name": "Demon Copperhead", "price": 850, "features": ["Barbara Kingsolver", "Pulitzer Prize", "Modern Classic"]}
            ],
            "Finance": [
                {"name": "The Psychology of Money", "price": 349, "features": ["Morgan Housel", "Wealth & Behavior"]},
                {"name": "Rich Dad Poor Dad", "price": 299, "features": ["Robert Kiyosaki", "Financial Literacy"]},
                {"name": "The Intelligent Investor", "price": 1200, "features": ["Benjamin Graham", "Value Investing", "Investing Bible"]},
                {"name": "A Random Walk Down Wall Street", "price": 950, "features": ["Burton Malkiel", "Investment Guide", "Market Analysis"]}
            ]
        },
        "beauty": {
            "Skin Care": [
                {"name": "Luxury Anti-Aging Cream", "price": 2500, "features": ["Retinol Infused", "Reduces Wrinkles", "Hydrating"]},
                {"name": "Hyaluronic Acid Serum", "price": 850, "features": ["Intense Hydration", "Plumps Skin", "All Skin Types"]},
                {"name": "Vitamin C Serum", "price": 699, "features": ["20% Vitamin C", "Brightens skin"]},
                {"name": "Sunscreen SPF 50+", "price": 599, "features": ["Matte finish", "Water-resistant"]},
                {"name": "Foaming Face Wash", "price": 350, "features": ["Gentle Cleansing", "Removes Impurities", "Daily Use"]}
            ],
            "Makeup": [
                {"name": "Premium Eyeshadow Palette", "price": 1500, "features": ["12 Shades", "Highly Pigmented", "Long-lasting"]},
                {"name": "Liquid Foundation", "price": 1100, "features": ["Full Coverage", "Matte Finish", "SPF 25"]},
                {"name": "Matte Lipstick Set", "price": 999, "features": ["Set of 5", "Long-lasting"]},
                {"name": "Kajal & Eyeliner Combo", "price": 299, "features": ["Smudge-proof", "Intense Black"]},
                {"name": "Blush & Highlighter Duo", "price": 650, "features": ["Natural Glow", "Buildable Color", "Compact"]}
            ]
        }
    }

    category_lower = category.lower()
    if category_lower not in product_catalog:
        return "Sorry, we don't have recommendations in that category."

    output = f"Top picks in {category.title()} "
    if budget_preference != "N/A":
        output += f"(within '{budget_preference}' budget) "
    if urgency_preference != "N/A" and urgency_preference != "just Browse":
        output += f"(for purchase '{urgency_preference}') "
    output += ":\n\n"

    found_recommendations = False

    for subcat, items in product_catalog[category_lower].items():
        filtered_items = []
        for p in items:
            # Apply budget filtering
            price = p["price"]
            passes_budget = False
            if budget_preference == "under 500":
                if price <= 500: passes_budget = True
            elif budget_preference == "500 to 1000":
                if 500 < price <= 1000: passes_budget = True
            elif budget_preference == "above 1000":
                if price > 1000: passes_budget = True
            elif budget_preference == "N/A" or budget_preference == "": 
                passes_budget = True

            if passes_budget:
                filtered_items.append(p)
        
        if filtered_items:
            output += f"**{subcat}**\n"
            for p in filtered_items:
                features = ", ".join(p["features"])
                output += f"• {p['name']} – ₹{p['price']:,}\n   Features: {features}\n\n"
            output += "\n" 
            found_recommendations = True
    
    if not found_recommendations:
        output = f"Sorry, we couldn't find products in {category.title()} matching your specified preferences. Try adjusting your selections or explore other categories."

    return output


# Lead magnet distributor 
def offer_resources_gradio(choice_label, email_address):
    resources = {
        "1": "Beginner's Guide to Online Shopping (PDF)",
        "2": "2025 Electronics Buying Trends (PDF)",
        "3": "How to Save Money While Shopping Online (PDF)"
    }
    
    choice = choice_label.split('.')[0].strip() 
    
    if not email_address or "@" not in email_address:
        return "Please provide a valid email address to receive the resource."

    if choice in resources:
        resource_name = resources[choice]
        return f"Great choice! '{resource_name}' has been sent to your email: **{email_address}**. Check your inbox!"
    else:
        return "Invalid selection."

# Demo booking
def book_demo_gradio(choice_label, email_address):
    slots = ["Mon 11 AM", "Tue 3 PM", "Wed 6 PM", "Fri 2 PM"]
    try:
        slot_index = int(choice_label.split('.')[0].strip()) - 1
        if 0 <= slot_index < len(slots):
            slot = slots[slot_index]
            if not email_address or "@" not in email_address:
                return "Please provide a valid email address to confirm your booking."
            return f"Meeting booked for **{slot}**. A confirmation has been sent to your email: **{email_address}**!"
        else:
            return "Invalid selection."
    except (ValueError, IndexError):
        return "Invalid selection."


# --- Gradio Chatbot Logic for Contact Details ---

def process_contact_input(message, history, state):
    # Initialize state for the contact details flow if it's the first turn or if reset
    if not state or 'current_question_index' not in state or state.get('reset_chat'):
        state = {
            'current_question_index': 0,
            'temp_answers': {},
            'history': []
        }
        # Clear the reset_chat flag
        state['reset_chat'] = False
        # Start with the first question
        first_question_key = QUALIFICATION_QUESTION_KEYS[state['current_question_index']]
        bot_message = QUESTIONS[first_question_key]
        # Ensure the first message from the bot is appended correctly
        return history + [[None, bot_message]], state

    current_idx = state['current_question_index']
    temp_answers = state['temp_answers']

    # Store the user's previous answer if a message was provided
    if message is not None and message.strip() != "":
        previous_question_key = QUALIFICATION_QUESTION_KEYS[current_idx - 1] 
        temp_answers[previous_question_key] = message.strip()
        history.append([message, None]) 

    # Move to the next question or finalize
    if current_idx < len(QUALIFICATION_QUESTION_KEYS) - 1:
        state['current_question_index'] += 1
        next_question_key = QUALIFICATION_QUESTION_KEYS[state['current_question_index']]
        bot_message = QUESTIONS[next_question_key]
        history.append([None, bot_message]) 
        return history, state
    else:
        if message is not None and message.strip() != "":
             last_question_key = QUALIFICATION_QUESTION_KEYS[current_idx]
             temp_answers[last_question_key] = message.strip()
        
        final_result = qualify_and_score_lead_internal(temp_answers)
        bot_message = final_result + "<br><br>Qualification complete. Click 'Back to Main Menu' to proceed."
        history.append([None, bot_message]) # Add final result to history
        
        # Mark state for reset on next entry to this block
        state = {
            'current_question_index': 0,
            'temp_answers': {},
            'reset_chat': True 
        }
        return history, state

# --- Function to read CSV for Dashboard ---
def read_leads_csv():
    try:
        df = pd.read_csv("shopping_leads.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["name", "email", "product_interest", "budget", "urgency", "intent", "score", "segment", "timestamp"])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["name", "email", "product_interest", "budget", "urgency", "intent", "score", "segment", "timestamp"])

# --- Function to read CSV and provide dashboard data, summaries, and plots ---
def get_dashboard_data():
    df = read_leads_csv()
    
    summary_markdown = ""
    segment_plot_fig = plt.figure() 
    product_plot_fig = plt.figure() 

    if df.empty:
        summary_markdown = "No lead data available yet. Please submit contact details first."
        return pd.DataFrame(), summary_markdown, segment_plot_fig, product_plot_fig

    total_leads = len(df)
    hot_leads = len(df[df['segment'] == 'Hot Lead'])
    warm_leads = len(df[df['segment'] == 'Warm Lead'])
    cold_leads = len(df[df['segment'] == 'Cold Lead'])

    product_interest_counts = df['product_interest'].value_counts()
    top_product_interests_text = ""
    if not product_interest_counts.empty:
        top_product_interests_list = [
            f"{idx}: {count}" for idx, count in product_interest_counts.head(3).items()
        ]
        top_product_interests_text = "Top Product Interests:<br>" + "<br>".join(top_product_interests_list)
    
    summary_markdown = f"""
    ### Lead Summary:
    - Total Leads: **{total_leads}**
    - Hot Leads: **{hot_leads}**
    - Warm Leads: **{warm_leads}**
    - Cold Leads: **{cold_leads}**
    <br>
    {top_product_interests_text}
    """

    # --- Generate Plots ---
    # Lead Segment Distribution Pie Chart
    segment_counts = df['segment'].value_counts()
    if not segment_counts.empty:
        segment_plot_fig = plt.figure(figsize=(6, 6))
        plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title('Lead Segment Distribution')
        plt.axis('equal') 
    else:
        segment_plot_fig = plt.figure()

    # Top Product Interests Bar Chart
    if not product_interest_counts.empty:
        product_plot_fig = plt.figure(figsize=(8, 5))
        product_interest_counts.head(5).plot(kind='bar', color='skyblue')
        plt.title('Top 5 Product Interests')
        plt.xlabel('Product Interest')
        plt.ylabel('Number of Leads')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    else:
        product_plot_fig = plt.figure()

    return df, summary_markdown, segment_plot_fig, product_plot_fig

def submit_contact_details(name, email, product_interest_selected, budget_selected, urgency_selected, intent_selected):
    """Collects all form inputs and passes them to the lead qualification logic."""
    temp_answers = {
        "name": name,
        "email": email,
        "product_interest": product_interest_selected,
        "budget": budget_selected,
        "urgency": urgency_selected,
        "intent": intent_selected
    }
    qualify_and_score_lead_internal(temp_answers)
    return "Contact details updated!"

def authenticate_and_prepare_dashboard(username, password):
    """
    Authenticates user and prepares dashboard data if successful.
    Returns updates for login_block, dashboard_block, input fields,
    login message, and dashboard data including plots.
    """
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        dashboard_df, dashboard_summary, segment_plot_fig, product_plot_fig = get_dashboard_data()
        return [
            gr.update(visible=False), 
            gr.update(visible=True),  
            gr.update(value=""),      
            gr.update(value=""),      
            gr.update(value="Login successful!", visible=True), 
            dashboard_df,             
            dashboard_summary,        
            segment_plot_fig,         
            product_plot_fig         
        ]
    else:
        return [
            gr.update(visible=True),  
            gr.update(visible=False), 
            gr.update(value=username),
            gr.update(value=""),      
            gr.update(value="Invalid username or password.", visible=True), 
            pd.DataFrame(),           
            "",                       
            plt.figure(),             
            plt.figure()              
        ]

# --- Gradio Interface Setup ---

with gr.Blocks(
    title="ShopSmart Assistant",
    css="""
        /* Ensure the main Gradio container doesn't get hidden under the sticky header */
        .gradio-container {
            padding-top: var(--header-height, 0px); /* This will be set dynamically by Gradio's internal CSS */
        }
        /* Style the fixed header row */
        #fixed-header {
            position: sticky;
            top: 0;
            width: 100%;
            z-index: 1000; /* Ensures it stays on top of other content */
            background-color: var(--background-fill-primary); /* Match Gradio's background */
            border-bottom: 1px solid var(--border-color-primary); /* Optional: add a subtle border */
        }
        /* Make plots responsive */
        .gradio-plot {
            width: 100% !important;
            height: auto !important;
        }
    """
) as demo:
    login_status_message = gr.Markdown("", visible=False) 

    with gr.Row(variant="panel", elem_id="fixed-header"):
        with gr.Column(scale=8):
            gr.Markdown("## ShopSmart Assistant") 
        with gr.Column(scale=2, min_width=100):
            global_login_btn = gr.Button("Admin Login")

    main_menu_block = gr.Group(visible=True)
    contact_block = gr.Group(visible=False)
    faq_block = gr.Group(visible=False)
    product_block = gr.Group(visible=False)
    resources_block = gr.Group(visible=False)
    demo_block = gr.Group(visible=False)
    thank_you_block = gr.Group(visible=False)
    login_block = gr.Group(visible=False) 
    dashboard_block = gr.Group(visible=False) 

    global_login_btn.click(
        lambda: [
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=True),  
            gr.update(value="", visible=False) 
        ],
        outputs=[
            main_menu_block, contact_block, faq_block, product_block, resources_block, demo_block, thank_you_block,
            dashboard_block,
            login_block,
            login_status_message 
        ],
        show_progress=False
    )
    
    with main_menu_block:
        gr.Markdown("## Main Menu") 
        gr.Markdown("What can I help you with today?")
        btn1 = gr.Button("Contact Details for Updates, Offers & Support") 
        btn2 = gr.Button("I need help with a question (FAQs)")
        btn3 = gr.Button("Recommend a product")
        btn4 = gr.Button("Send me a free guide")
        btn5 = gr.Button("Book a meeting/demo")
        gr.Markdown("---") 
        btn0 = gr.Button("Exit") 

    # --- Contact Details Block ---
    with contact_block:
        gr.Markdown("## Contact Details for Updates, Offers & Support")
        gr.Markdown("Please fill out the details below:") 
        name_input = gr.Textbox(label="Your name:", placeholder="e.g., John Doe")
        email_input = gr.Textbox(label="Your email:", placeholder="e.g., john.doe@example.com")
        
        product_input = gr.Dropdown(
            label=QUESTIONS["product_interest"],
            choices=LABELS["product_interest"],
            value=None, 
            allow_custom_value=True 
        )
        budget_input = gr.Dropdown(
            label=QUESTIONS["budget"],
            choices=LABELS["budget"],
            value=None
        )
        urgency_input = gr.Dropdown(
            label=QUESTIONS["urgency"],
            choices=LABELS["urgency"],
            value=None
        )
        intent_input = gr.Dropdown(
            label=QUESTIONS["intent"],
            choices=LABELS["intent"],
            value=None
        )

        submit_btn_contact = gr.Button("Submit Details") 
        contact_output_markdown = gr.Markdown() 

        submit_btn_contact.click(
            submit_contact_details, 
            inputs=[
                name_input, email_input,
                product_input, budget_input, urgency_input, intent_input
            ],
            outputs=contact_output_markdown,
            show_progress=True 
        )

        gr.Button("Back to Main Menu").click(
            lambda: [
                gr.update(visible=True), gr.update(visible=False), 
                gr.update(value=""),      
                gr.update(value=""),      
                gr.update(value=None),    
                gr.update(value=None),    
                gr.update(value=None),    
                gr.update(value=None),    
                gr.update(value="")       
            ],
            outputs=[
                main_menu_block, contact_block,
                name_input, email_input, product_input, budget_input, urgency_input, intent_input,
                contact_output_markdown
            ],
            show_progress=False
        )

    # --- FAQ Block ---
    with faq_block:
        gr.Markdown("## I have a question (FAQs)")
        gr.Markdown("""
        I can help with common questions about shipping, returns, payments, and more!
        """)
        faq_chat = gr.ChatInterface(
            faq_chat_interface,
            chatbot=gr.Chatbot(label="FAQ Chat", height=300, type="messages"),
            textbox=gr.Textbox(placeholder="Ask your question (or type 'exit' to leave FAQs):"),
            submit_btn="Ask",
            autoscroll=True
        )
        gr.Button("Back to Main Menu").click(
            lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(value=[])],
            outputs=[main_menu_block, faq_block, faq_chat.chatbot],
            show_progress=False
        )

    # --- Product Recommendations Block ---
    with product_block:
        gr.Markdown("## Recommend a product")
        category_input = gr.Textbox(label="What product category are you interested in? (electronics, fashion, books, beauty):")
        
        budget_input_recommend = gr.Dropdown(
            label="What's your estimated budget?",
            choices=LABELS["budget"] + ["N/A"], 
            value="N/A"
        )
        urgency_input_recommend = gr.Dropdown(
            label="When are you planning to buy?",
            choices=LABELS["urgency"] + ["N/A"], 
            value="N/A"
        )

        recommend_btn = gr.Button("Get Recommendations")
        output_text_product = gr.Markdown()
        
        recommend_btn.click(
            recommend_products_gradio,
            inputs=[category_input, budget_input_recommend, urgency_input_recommend], 
            outputs=output_text_product
        )
        gr.Button("Back to Main Menu").click(
            lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(value=""), gr.update(value="N/A"), gr.update(value="N/A")],
            outputs=[main_menu_block, product_block, category_input, budget_input_recommend, urgency_input_recommend],
            show_progress=False
        )

    # --- Resources Block ---
    with resources_block:
        gr.Markdown("## Available Resources:")
        resources_options = [
            ("1. Beginner's Guide to Online Shopping (PDF)", "1"),
            ("2. 2025 Electronics Buying Trends (PDF)", "2"),
            ("3. How to Save Money While Shopping Online (PDF)", "3")
        ]
        choice_radio_resources = gr.Radio(choices=[opt[0] for opt in resources_options], label="Choose one to download:", type="value")
        email_input_resources = gr.Textbox(label="Enter your email to receive the guide:", placeholder="your.email@example.com")
        download_btn = gr.Button("Download")
        output_text_resources = gr.Markdown()
        
        download_btn.click(
            offer_resources_gradio,
            inputs=[choice_radio_resources, email_input_resources], 
            outputs=output_text_resources
        )
        gr.Button("Back to Main Menu").click(
            lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(value=""), ""],
            outputs=[main_menu_block, resources_block, choice_radio_resources, email_input_resources, output_text_resources],
            show_progress=False
        )

    # --- Book Demo Block ---
    with demo_block:
        gr.Markdown("## Book a meeting/demo")
        slots_options = [
            ("1. Mon 11 AM", "1"),
            ("2. Tue 3 PM", "2"),
            ("3. Wed 6 PM", "3"),
            ("4. Fri 2 PM", "4")
        ]
        choice_radio_demo = gr.Radio(choices=[opt[0] for opt in slots_options], label="Available Time Slots:", type="value")
        email_input_demo = gr.Textbox(label="Enter your email for booking confirmation:", placeholder="your.email@example.com")
        book_btn = gr.Button("Confirm Booking")
        output_text_demo = gr.Markdown()

        book_btn.click(
            book_demo_gradio,
            inputs=[choice_radio_demo, email_input_demo], 
            outputs=output_text_demo
        )
        gr.Button("Back to Main Menu").click(
            lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(value=""), ""],
            outputs=[main_menu_block, demo_block, choice_radio_demo, email_input_demo, output_text_demo],
            show_progress=False
        )

    # ---User Dashboard Block ---
    with dashboard_block:
        gr.Markdown("## User Dashboard")

        summary_output_markdown = gr.Markdown("Loading summary...") 

        lead_segment_plot = gr.Plot(label="Lead Segment Distribution")
        top_products_plot = gr.Plot(label="Top Product Interests")

        dashboard_data = gr.Dataframe(
            value=read_leads_csv, 
            headers=["Name", "Email", "Product Interest", "Budget", "Urgency", "Intent", "Score", "Segment", "Timestamp"],
            row_count=(5, "dynamic"),
            col_count=(9, "fixed"),
            wrap=True,
            interactive=False,
        )
        refresh_btn = gr.Button("Refresh Data")

        with gr.Row():
            with gr.Column(scale=1):
                summary_output_markdown
            with gr.Column(scale=2):
                lead_segment_plot
        with gr.Row():
            top_products_plot
        with gr.Row():
            dashboard_data

        refresh_btn.click(
            get_dashboard_data,
            inputs=[],
            outputs=[dashboard_data, summary_output_markdown, lead_segment_plot, top_products_plot]
        )

        logout_btn = gr.Button("Logout")
        logout_btn.click(
            lambda: [
                gr.update(visible=True), 
                gr.update(visible=False),
                gr.update(value=pd.DataFrame()), 
                gr.update(value=""), 
                gr.update(value=plt.figure()), 
                gr.update(value=plt.figure())  
            ],
            outputs=[main_menu_block, dashboard_block, dashboard_data, summary_output_markdown, lead_segment_plot, top_products_plot],
            show_progress=False
        )

    # --- Thank You Block ---
    with thank_you_block:
        gr.Markdown("## Thank You for chatting with us!")
        gr.Markdown("We appreciate your time. Have a great day!")

    # --- Login Block Definition ---
    with login_block:
        gr.Markdown("## Dashboard Login")
        username_input = gr.Textbox(label="Username", placeholder="Enter admin username")
        password_input = gr.Textbox(label="Password", type="password", placeholder="Enter admin password")
        login_btn = gr.Button("Login")

        login_btn.click(
            fn=authenticate_and_prepare_dashboard,
            inputs=[username_input, password_input],
            outputs=[
                login_block,          
                dashboard_block,     
                username_input,       
                password_input,       
                login_status_message, 
                dashboard_data,       
                summary_output_markdown, 
                lead_segment_plot,    
                top_products_plot     
            ],
            show_progress=True
        )

    with login_block:
        gr.Button("Back to Main Menu").click(
            lambda: [
                gr.update(visible=True), 
                gr.update(visible=False),
                gr.update(value=""),     
                gr.update(value=""),     
                gr.update(value="", visible=False) 
            ],
            outputs=[
                main_menu_block, login_block,
                username_input, password_input, login_status_message
            ],
            show_progress=False
        )


    # ---  Main Menu Button Actions ---
    btn1.click(
        lambda: [gr.update(visible=False), gr.update(visible=True)], 
        outputs=[main_menu_block, contact_block], 
        show_progress=False
    )

    btn2.click(lambda: [gr.update(visible=False), gr.update(visible=True)], outputs=[main_menu_block, faq_block], show_progress=False)
    btn3.click(lambda: [gr.update(visible=False), gr.update(visible=True)], outputs=[main_menu_block, product_block], show_progress=False)
    btn4.click(lambda: [gr.update(visible=False), gr.update(visible=True)], outputs=[main_menu_block, resources_block], show_progress=False)
    btn5.click(lambda: [gr.update(visible=False), gr.update(visible=True)], outputs=[main_menu_block, demo_block], show_progress=False)
    
    btn0.click(
        lambda: [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True)
        ],
        outputs=[
            main_menu_block, 
            contact_block, 
            faq_block, 
            product_block, 
            resources_block, 
            demo_block, 
            dashboard_block,
            thank_you_block
        ],
        show_progress=False
    )


# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
