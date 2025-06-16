# Lead Qualification Chatbot

The application features:
* **Automated Lead Qualification:** Intelligently collects and processes customer information to assign lead scores and segments.
* **Enhanced Customer Interaction:** Provides a multi-functional chatbot for FAQs, product recommendations, resource distribution, and demo scheduling.
* **Intuitive Admin Dashboard:** Offers secure access to detailed lead data, summarized insights, and interactive visualizations for better sales strategy.

## Features

* **Lead Generation & Qualification:**
    * **Contact Details Collection:** Gathers essential customer information (name, email, product interest, budget, urgency, intent).
    * **Automated Scoring & Segmentation:** Utilizes a zero-shot classification model and a custom scoring logic to assign a numerical lead score and classify leads into "Hot," "Warm," or "Cold" segments.
    * **Data Persistence:** All qualified lead data is saved to a `shopping_leads.csv` file for easy access and analysis.
* **Customer Engagement & Support:**
    * **FAQ Chatbot:** Answering common customer questions (shipping, returns, payments, support) powered by a local Retrieval-Augmented Generation (RAG) system using Ollama and LangChain.
    * **Personalized Product Recommendations:** Suggests relevant products based on the customer's stated interests, budget, and urgency, drawing from a predefined product catalog.
    * **Resource Distribution:** Offers valuable digital resources (e.g., guides, trend reports) to interested customers via email.
    * **Demo/Meeting Booking:** Facilitates easy scheduling of meetings or product demos with potential customers.
* **Admin Dashboard:**
    * **Secure Access:** Protected by a customizable admin username and password.
    * **Comprehensive Lead Overview:** Displays all collected lead data in an interactive and filterable table (`gr.Dataframe`).
    * **Key Performance Indicators (KPIs):** Shows vital lead statistics such as total leads, and counts for hot, warm, and cold leads.
    * **Dynamic Visualizations:** Presents clear, interactive charts generated with Matplotlib for:
        * **Lead Segment Distribution:** A pie chart showing the percentage breakdown of Hot, Warm, and Cold leads.
        * **Top Product Interests:** A bar chart illustrating the most popular product categories among leads.
    * **Real-time Updates:** "Refresh Data" button to update the dashboard with the latest lead information.
      
## Prerequisites

* Install Ollama Models

   The FAQ chatbot relies on a local LLM provided by Ollama. Before running the application, you need to pull the specific model used in the    script. This project is configured to use llama3.2.

