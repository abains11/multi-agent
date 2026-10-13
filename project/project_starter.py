import pandas as pd
import numpy as np
import ast
import json
import re
import sys
import time
from pathlib import Path
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sqlalchemy import create_engine, Engine
from smolagents import (
    ChatMessage,
    ChatMessageToolCall,
    MessageRole,
    Model,
    Tool,
    ToolCallingAgent,
    tool,
)
from smolagents.monitoring import LogLevel
from smolagents.models import ChatMessageToolCallFunction

PROJECT_DIR = Path(__file__).resolve().parent

# Keep data and the SQLite file anchored to this script, not the shell's cwd.
db_engine = create_engine(f"sqlite:///{(PROJECT_DIR / 'munder_difflin.db').as_posix()}")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine = db_engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
            "cash_date": [],         # ISO-formatted date when cash changes
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv(PROJECT_DIR / "quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(PROJECT_DIR / "quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            metadata = [
                ast.literal_eval(value) if isinstance(value, str) else value
                for value in quotes_df["request_metadata"]
            ]
            quotes_df.loc[:, "request_metadata"] = metadata
            quotes_df.loc[:, "job_type"] = [value.get("job_type", "") for value in metadata]
            quotes_df.loc[:, "order_size"] = [value.get("order_size", "") for value in metadata]
            quotes_df.loc[:, "event_type"] = [value.get("event_type", "") for value in metadata]

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
            "cash_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
                "cash_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
    cash_date: Optional[Union[str, datetime]] = None,
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, inventory transaction date, and cash date into
    the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.
        cash_date (str or datetime, optional): Date cash changes. Defaults to `date`.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date
        cash_date_str = (
            cash_date.isoformat() if isinstance(cash_date, datetime)
            else cash_date or date_str
        )

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        with db_engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    INSERT INTO transactions
                        (item_name, transaction_type, units, price, transaction_date, cash_date)
                    VALUES
                        (:item_name, :transaction_type, :units, :price, :transaction_date, :cash_date)
                    """
                ),
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": price,
                    "transaction_date": date_str,
                    "cash_date": cash_date_str,
                },
            )
            return int(result.lastrowid)

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Cash can change on a different date from the associated stock movement.
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE cash_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


@tool
def inventory_snapshot_tool(as_of_date: str) -> Dict[str, int]:
    """Read the available inventory snapshot.

    Args:
        as_of_date: Inclusive inventory date in YYYY-MM-DD format.
    """
    return get_all_inventory(as_of_date)


@tool
def item_stock_tool(item_name: str, as_of_date: str) -> int:
    """Read the current stock for one catalog item.

    Args:
        item_name: Exact item name used by the catalog.
        as_of_date: Inclusive inventory date in YYYY-MM-DD format.
    """
    result = get_stock_level(item_name, as_of_date)
    return int(result["current_stock"].iloc[0])


@tool
def cash_balance_tool(as_of_date: str) -> float:
    """Read the company's cash balance on a specified date.

    Args:
        as_of_date: Inclusive balance date in YYYY-MM-DD format.
    """
    return get_cash_balance(as_of_date)


@tool
def financial_report_tool(as_of_date: str) -> Dict:
    """Generate the company's internal financial and inventory health report.

    Args:
        as_of_date: Inclusive report date in YYYY-MM-DD format.
    """
    return generate_financial_report(as_of_date)


@tool
def supplier_eta_tool(input_date_str: str, quantity: int) -> str:
    """Estimate a supplier delivery date using the starter's local lead-time rule.

    Args:
        input_date_str: Requested order date in YYYY-MM-DD format.
        quantity: Number of units to replenish.
    """
    return get_supplier_delivery_date(input_date_str, quantity)


@tool
def quote_history_tool(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """Find historical quotes related to the customer's request.

    Args:
        search_terms: Terms used to match customer requests and quote explanations.
        limit: Maximum number of historical quotes to return.
    """
    return search_quote_history(search_terms, limit)


@tool
def transaction_tool(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: str,
    cash_date: Optional[str] = None,
) -> int:
    """Record a stock purchase or completed sale in the SQLite ledger.

    Args:
        item_name: Exact catalog item name.
        transaction_type: Either stock_orders or sales.
        quantity: Number of units in this transaction.
        price: Total dollar amount for this transaction.
        date: Inventory transaction date in YYYY-MM-DD format.
        cash_date: Date cash changes; defaults to date.
    """
    return create_transaction(
        item_name,
        transaction_type,
        quantity,
        price,
        date,
        cash_date=cash_date,
    )


@tool
def transaction_analysis_tool(as_of_date: str) -> Dict:
    """Summarize completed inventory and sales transactions for the advisor.

    Args:
        as_of_date: Inclusive inventory transaction cutoff in YYYY-MM-DD format.
    """
    totals = pd.read_sql(
        """
        SELECT transaction_type, COUNT(*) AS transaction_count,
               COALESCE(SUM(units), 0) AS units,
               COALESCE(SUM(price), 0) AS amount
        FROM transactions
        WHERE transaction_date <= :as_of_date
          AND (transaction_type != 'sales' OR item_name IS NOT NULL)
        GROUP BY transaction_type
        """,
        db_engine,
        params={"as_of_date": as_of_date},
    )
    top_sales = pd.read_sql(
        """
        SELECT item_name, SUM(units) AS units, SUM(price) AS revenue
        FROM transactions
        WHERE transaction_type = 'sales'
          AND item_name IS NOT NULL
          AND transaction_date <= :as_of_date
        GROUP BY item_name
        ORDER BY revenue DESC
        LIMIT 5
        """,
        db_engine,
        params={"as_of_date": as_of_date},
    )
    catalog = pd.read_sql("SELECT item_name, min_stock_level FROM inventory", db_engine)
    low_stock = []
    for item in catalog.to_dict(orient="records"):
        stock = int(
            get_stock_level(str(item["item_name"]), as_of_date)["current_stock"].iloc[0]
        )
        if stock <= int(item["min_stock_level"]):
            low_stock.append(
                {
                    "item_name": str(item["item_name"]),
                    "stock": stock,
                    "minimum": int(item["min_stock_level"]),
                }
            )
    return {
        "transaction_totals": [
            {
                "transaction_type": str(row["transaction_type"]),
                "transaction_count": int(row["transaction_count"]),
                "units": int(row["units"]),
                "amount": round(float(row["amount"]), 2),
            }
            for row in totals.to_dict(orient="records")
        ],
        "top_sales": [
            {
                "item_name": str(row["item_name"]),
                "units": int(row["units"]),
                "revenue": round(float(row["revenue"]), 2),
            }
            for row in top_sales.to_dict(orient="records")
        ],
        "low_stock": low_stock,
    }


UNIT_WORDS = (
    r"(?:sheets?|reams?|rolls?|boxes?|packets?|packages?|packs?|pieces?|"
    r"units?|items?)"
)
REQUEST_ITEM_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9.])(?P<quantity>\d[\d,]*)\s*"
    rf"(?:(?:{UNIT_WORDS})\s+)?(?:of\s+)?"
    r"(?P<description>[A-Za-z0-9][A-Za-z0-9%\s,()\"'’×x./-]*?)"
    r"(?=(?:,\s*\d[\d,]*|,\s*along\s+with\s+\d[\d,]*|"
    r"\s+(?:and|along\s+with)\s+\d[\d,]*|"
    r"\n|;\s*\d[\d,]*|\s+for\s+(?:the|our|an|a|upcoming)\b|"
    r"\s+by\s+(?:the\s+)?(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\b|\.\s|$))",
    re.IGNORECASE,
)
DEADLINE_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)


def parse_requested_items(request: str) -> List[Dict[str, Union[str, int]]]:
    """Extract quantities and item descriptions from the supplied text requests."""
    items = []
    for match in REQUEST_ITEM_PATTERN.finditer(request):
        description = match.group("description").strip(" \t\r\n,.;:-")
        description = re.sub(r"\s+", " ", description)
        if description and not re.match(r"20\d{2}\b", description):
            items.append(
                {
                    "quantity": int(match.group("quantity").replace(",", "")),
                    "description": description,
                }
            )
    return items


def resolve_catalog_item(description: str) -> Tuple[Optional[str], str]:
    """Map requested wording to a catalog item without inventing stock variants."""
    value = description.lower().replace("×", "x")
    value = re.sub(r"\s+", " ", value)

    size = re.search(r"\bA[35]\b", description, re.IGNORECASE)
    if size:
        return None, f"{size.group(0).upper()} size is not represented in the catalog"
    if "biodegradable" in value:
        return None, "the catalog does not specify biodegradable product variants"
    if "recycled" in value and "cardstock" in value:
        return None, "the catalog has no recycled-cardstock variant"
    if "kraft" in value and "envelope" in value:
        return None, "the catalog does not identify kraft or recycled envelope variants"
    if any(term in value for term in ("cardboard", "balloon", "ticket")):
        return None, "the requested product is not in the catalog"

    if "washi" in value:
        return "Decorative adhesive tape (washi tape)", ""
    if "streamer" in value:
        return "Party streamers", ""
    if "flyer" in value:
        return "Flyers", ""
    if "poster board" in value:
        return "Large poster paper (24x36 inches)", ""
    if "poster" in value:
        if "220 gsm" in value:
            return "220 gsm poster paper", ""
        return None, "poster size or paper weight is not specific enough to match a catalog item"
    if "heavyweight" in value:
        return "Heavyweight paper", ""
    if "cardstock" in value or "card stock" in value:
        return "Cardstock", ""
    if "construction paper" in value:
        return "Construction paper", ""
    if "napkin" in value:
        return "Paper napkins", ""
    if "paper cup" in value or re.search(r"\bcups?\b", value):
        return "Paper cups", ""
    if "paper plate" in value or re.search(r"\bplates?\b", value):
        return "Paper plates", ""
    if "envelope" in value:
        return "Envelopes", ""
    if "glossy" in value:
        return "Glossy paper", ""
    if "matte" in value:
        return "Matte paper", ""
    if "recycled" in value:
        return "Recycled paper", ""
    if "colored" in value or "colour" in value:
        return "Colored paper", ""
    if "standard copy" in value or "standard printer" in value:
        return "Standard copy paper", ""
    if "printer paper" in value or "printing paper" in value:
        return "A4 paper" if "A4" in description.upper() else "Standard copy paper", ""
    if re.search(r"\bA4\b", description, re.IGNORECASE):
        return "A4 paper", ""

    catalog_names = {item["item_name"].lower(): item["item_name"] for item in paper_supplies}
    normalized = re.sub(r"[^a-z0-9 ]", " ", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for name, catalog_name in catalog_names.items():
        if name in normalized:
            return catalog_name, ""
    return None, "no matching item was found in the catalog"


def requested_deadline(request: str, request_date: str) -> str:
    match = DEADLINE_PATTERN.search(request)
    if not match:
        return request_date
    month, day, year = match.groups()
    return datetime.strptime(f"{month} {day} {year}", "%B %d %Y").strftime("%Y-%m-%d")


class InventoryAgent:
    """Own stock checks, restock feasibility, and inventory reservations."""

    def __init__(self) -> None:
        self.future_reservations: List[Dict[str, Union[str, int]]] = []
        self.future_reorder_costs: List[Dict[str, Union[str, float]]] = []

    def assess(
        self,
        requested_items: List[Dict[str, Union[str, int]]],
        request_date: str,
        deadline: str,
    ) -> Dict:
        inventory_snapshot_tool.forward(as_of_date=request_date)
        errors = []
        lines = []
        planned_reorders = []
        reserved_cash = sum(
            float(commitment["cost"])
            for commitment in self.future_reorder_costs
            if str(commitment["date"]) > request_date
        )
        available_cash = cash_balance_tool.forward(as_of_date=request_date) - reserved_cash

        if sum(int(item["quantity"]) for item in requested_items) >= 1000:
            financial_report_tool.forward(as_of_date=request_date)

        fulfillment_date = request_date
        for requested_item in requested_items:
            catalog_item, resolution_error = resolve_catalog_item(
                str(requested_item["description"])
            )
            if catalog_item is None:
                errors.append(f"{requested_item['description']}: {resolution_error}")
                continue

            quantity = int(requested_item["quantity"])
            current_stock = item_stock_tool.forward(
                item_name=catalog_item,
                as_of_date=request_date,
            )
            reserved_units = sum(
                int(reservation["quantity"])
                for reservation in self.future_reservations
                if reservation["item_name"] == catalog_item
                and str(reservation["date"]) > request_date
            )
            available_stock = max(0, current_stock - reserved_units)
            shortage = max(0, quantity - available_stock)
            reorder_date = request_date

            if shortage:
                reorder_date = supplier_eta_tool.forward(
                    input_date_str=request_date,
                    quantity=shortage,
                )
                if reorder_date > deadline:
                    errors.append(
                        f"{catalog_item}: {shortage:,} additional units cannot arrive "
                        f"by {deadline} (earliest estimated date {reorder_date})"
                    )
                    continue
                entry = next(
                    item for item in paper_supplies if item["item_name"] == catalog_item
                )
                planned_reorders.append(
                    {
                        "item_name": catalog_item,
                        "quantity": shortage,
                        "cost": round(shortage * float(entry["unit_price"]), 2),
                        "date": reorder_date,
                    }
                )
                available_stock += shortage

            fulfillment_date = max(fulfillment_date, reorder_date)
            entry = next(
                item for item in paper_supplies if item["item_name"] == catalog_item
            )
            lines.append(
                {
                    "item_name": catalog_item,
                    "quantity": quantity,
                    "unit_price": float(entry["unit_price"]),
                    "available_before_reorder": max(0, current_stock - reserved_units),
                    "stock_at_assessment": current_stock,
                }
            )

        reorder_total = round(sum(float(order["cost"]) for order in planned_reorders), 2)
        if reorder_total > available_cash:
            errors.append(
                f"restocking costs ${reorder_total:,.2f}, exceeding available cash "
                f"${max(0.0, available_cash):,.2f}"
            )
        return {
            "feasible": not errors,
            "errors": errors,
            "lines": lines,
            "reorders": planned_reorders if not errors else [],
            "fulfillment_date": fulfillment_date,
            "reorder_total": reorder_total,
            "available_cash": available_cash,
        }

    def commit_reorders(self, plan: Dict) -> None:
        for reorder in plan["reorders"]:
            transaction_tool.forward(
                item_name=str(reorder["item_name"]),
                transaction_type="stock_orders",
                quantity=int(reorder["quantity"]),
                price=float(reorder["cost"]),
                date=str(reorder["date"]),
            )
            if str(reorder["date"]) > str(plan["request_date"]):
                self.future_reorder_costs.append(
                    {"date": str(reorder["date"]), "cost": float(reorder["cost"])}
                )

    def reserve_order(self, plan: Dict) -> None:
        delivery_date = str(plan["fulfillment_date"])
        if delivery_date <= str(plan["request_date"]):
            return
        self.future_reservations.extend(
            {
                "item_name": str(line["item_name"]),
                "quantity": int(line["quantity"]),
                "date": delivery_date,
            }
            for line in plan["lines"]
        )


class QuoteAgent:
    """Own history lookups, catalog price calculation, and bulk discounts."""

    def create_quote(self, lines: List[Dict], customer_brief: Dict[str, str]) -> Dict:
        history = []
        for search_term in (
            customer_brief.get("event", ""),
            customer_brief.get("job", ""),
            customer_brief.get("need_size", ""),
        ):
            if search_term:
                history = quote_history_tool.forward(
                    search_terms=[search_term],
                    limit=5,
                )
                if history:
                    break

        subtotal = round(
            sum(float(line["unit_price"]) * int(line["quantity"]) for line in lines),
            2,
        )
        units = sum(int(line["quantity"]) for line in lines)
        discount_rate = (
            0.10 if units >= 5000 else 0.08 if units >= 1500 else 0.05 if units >= 500 else 0.0
        )
        discount = round(subtotal * discount_rate, 2)
        if discount_rate:
            price_rationale = (
                f"A {discount_rate:.0%} volume discount applies to the "
                f"{units:,}-unit request."
            )
        else:
            price_rationale = (
                f"The {units:,}-unit request is below the 500-unit volume-discount "
                "threshold, so catalog prices apply without a volume discount."
            )
        if customer_brief.get("price_sensitive") == "true":
            price_rationale += (
                " Your request asked for price-conscious options; the quote uses "
                "the available catalog prices and published volume tier."
            )
        return {
            "subtotal": subtotal,
            "discount_rate": discount_rate,
            "discount": discount,
            "total": round(subtotal - discount, 2),
            "price_rationale": price_rationale,
            "history_note": (
                f"Found {len(history)} historical quote(s) for context."
                if history
                else "No comparable historical quote was found; catalog prices were used."
            ),
        }


class OrderFulfillmentAgent:
    """Own accepted-order validation and sale transactions."""

    def complete(
        self,
        plan: Dict,
        quote: Dict,
        request_date: str,
        inventory: InventoryAgent,
    ) -> int:
        for line in plan["lines"]:
            stock = item_stock_tool.forward(
                item_name=str(line["item_name"]),
                as_of_date=request_date,
            )
            if stock < int(line["stock_at_assessment"]):
                raise RuntimeError(
                    f"Inventory changed during fulfillment for {line['item_name']}."
                )

        inventory.commit_reorders(plan)
        remaining_discount = float(quote["discount"])
        lines = plan["lines"]
        for index, line in enumerate(lines):
            if index == len(lines) - 1:
                line_discount = remaining_discount
            else:
                fraction = (
                    float(line["unit_price"]) * int(line["quantity"])
                    / float(quote["subtotal"])
                    if quote["subtotal"]
                    else 0.0
                )
                line_discount = round(float(quote["discount"]) * fraction, 2)
                remaining_discount = round(remaining_discount - line_discount, 2)
            transaction_tool.forward(
                item_name=str(line["item_name"]),
                transaction_type="sales",
                quantity=int(line["quantity"]),
                price=round(
                    float(line["unit_price"]) * int(line["quantity"]) - line_discount,
                    2,
                ),
                date=str(plan["fulfillment_date"]),
                cash_date=request_date,
            )
        inventory.reserve_order(plan)
        return len(lines)


class BusinessAdvisorAgent:
    """Turn internal transaction and inventory trends into recommendations."""

    def analyze(self, as_of_date: str, evaluation_results: List[Dict]) -> Dict:
        analysis = transaction_analysis_tool.forward(as_of_date=as_of_date)
        fulfilled_count = sum(bool(result["accepted"]) for result in evaluation_results)
        rejected_results = [
            result for result in evaluation_results if result["status"] == "rejected"
        ]
        catalog_rejections = sum(
            "catalog" in str(result["response"]).lower()
            or "represented" in str(result["response"]).lower()
            for result in rejected_results
        )
        delivery_rejections = sum(
            "requested delivery date" in str(result["response"]).lower()
            for result in evaluation_results
        )

        recommendations = []
        if analysis["low_stock"]:
            low_stock_names = ", ".join(
                f"{item['item_name']} ({item['stock']} on hand; minimum {item['minimum']})"
                for item in analysis["low_stock"]
            )
            recommendations.append(
                "Review reorder points for these items before accepting more large "
                f"orders: {low_stock_names}."
            )
        if analysis["top_sales"]:
            best_seller = analysis["top_sales"][0]
            recommendations.append(
                f"Prioritize replenishment planning for {best_seller['item_name']}, "
                f"the leading recorded seller ({best_seller['units']:,} units; "
                f"${best_seller['revenue']:,.2f} revenue)."
            )
        if catalog_rejections:
            recommendations.append(
                f"{catalog_rejections} sample requests were rejected for unsupported "
                "catalog specifications. Review demand before adding variants, and "
                "capture size/weight explicitly to avoid unfulfillable quotes."
            )
        if delivery_rejections:
            recommendations.append(
                f"{delivery_rejections} sample requests missed stated deadlines under "
                "the local lead-time estimates. Compare actual supplier lead times "
                "and consider offering a confirmed later delivery option."
            )
        if not recommendations:
            recommendations.append(
                "No low-stock or rejected-request signal was found; continue "
                "monitoring sales mix and replenishment timing."
            )

        return {
            "as_of_date": as_of_date,
            "accepted_requests": fulfilled_count,
            "rejected_requests": len(rejected_results),
            "transaction_totals": analysis["transaction_totals"],
            "top_sales": analysis["top_sales"],
            "low_stock": analysis["low_stock"],
            "recommendations": recommendations,
        }


class BusinessActionTool(Tool):
    """Expose a specialist's deterministic business actions as a framework tool."""

    name = "run_business_action"
    description = "Execute one validated deterministic action for this specialist."
    inputs = {
        "task_json": {
            "type": "string",
            "description": "JSON object containing the action name and its arguments.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        handler: Callable[[str, Dict[str, Any]], Any],
    ) -> None:
        super().__init__()
        self.handler = handler

    def forward(self, task_json: str) -> str:
        task = json.loads(task_json)
        if not isinstance(task, dict):
            raise ValueError("Business action payload must be a JSON object.")
        action = task.get("action")
        arguments = task.get("arguments")
        if not isinstance(action, str) or not isinstance(arguments, dict):
            raise ValueError("Business action payload requires an action and arguments object.")
        return json.dumps(self.handler(action, arguments))


class DeterministicWorkflowModel(Model):
    """Translate structured workflow requests into smolagents tool calls locally."""

    def __init__(self) -> None:
        super().__init__(model_id="local-deterministic-workflow")
        self.call_number = 0

    @staticmethod
    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        return "" if content is None else str(content)

    def _tool_call(self, name: str, arguments: Dict[str, Any]) -> ChatMessage:
        self.call_number += 1
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=[
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name=name,
                        arguments=arguments,
                    ),
                    id=f"workflow-call-{self.call_number}",
                    type="function",
                )
            ],
        )

    def generate(
        self,
        messages: List[ChatMessage],
        tools_to_call_from: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> ChatMessage:
        unsupported_options = set(kwargs) - {"stop_sequences"}
        if unsupported_options:
            raise ValueError(
                "Unsupported deterministic-model options: "
                + ", ".join(sorted(unsupported_options))
            )
        tool_options = {tool.name: tool for tool in (tools_to_call_from or [])}
        for message in reversed(messages):
            if getattr(message.role, "value", message.role) == MessageRole.TOOL_RESPONSE.value:
                observation = self._text(message.content)
                if observation.startswith("Observation:\n"):
                    observation = observation[len("Observation:\n") :]
                return self._tool_call(
                    "final_answer",
                    {"answer": observation},
                )

        user_message = next(
            (
                message
                for message in reversed(messages)
                if getattr(message.role, "value", message.role) == MessageRole.USER.value
            ),
            None,
        )
        if user_message is None:
            raise RuntimeError("Framework agent received no workflow task.")

        task_text = self._text(user_message.content)
        if task_text.startswith("New task:\n"):
            task_text = task_text[len("New task:\n") :]
        try:
            request = json.loads(task_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Framework agent task must be a JSON workflow request.") from exc
        if not isinstance(request, dict):
            raise ValueError("Framework agent task must be a JSON object.")

        if "agent" in request:
            agent_name = request["agent"]
            if not isinstance(agent_name, str) or agent_name not in tool_options:
                raise ValueError(f"Unknown managed specialist: {agent_name!r}.")
            task = request.get("task")
            if not isinstance(task, str):
                raise ValueError("Managed specialist requests require a string task.")
            return self._tool_call(
                agent_name,
                {"task": task, "additional_args": None},
            )

        specialist_tool = "run_business_action"
        if specialist_tool not in tool_options:
            raise ValueError("The specialist has no business-action tool registered.")
        return self._tool_call(
            specialist_tool,
            {"task_json": task_text},
        )


class TerminalProgressDisplay:
    """Show a compact staged progress animation only in an interactive terminal."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.completed_steps = 0
        self.total_steps = 4

    def update(self, stage: str, detail: str) -> None:
        if not self.enabled:
            return
        self.completed_steps = min(self.completed_steps + 1, self.total_steps)
        progress = int(12 * self.completed_steps / self.total_steps)
        bar = "#" * progress + "-" * (12 - progress)
        line = f"[{bar}] {stage}: {detail}"
        for frame in ("-", "\\", "|"):
            print(f"\r\033[2K[{frame}{bar[1:]}] {stage}: working...", end="", flush=True)
            time.sleep(0.035)
        print(f"\r\033[2K{line}", flush=True)


class ProjectOrchestrator:
    """Coordinate specialists and advocate for explicit customer requirements."""

    def __init__(self) -> None:
        self.inventory_agent = InventoryAgent()
        self.quote_agent = QuoteAgent()
        self.order_agent = OrderFulfillmentAgent()
        self.business_advisor = BusinessAdvisorAgent()
        self.workflow_model = DeterministicWorkflowModel()
        self.inventory_worker = self._create_specialist(
            "inventory_specialist",
            "Checks stock, estimates replenishment, and evaluates cash feasibility.",
            self._run_inventory_action,
        )
        self.quote_worker = self._create_specialist(
            "quote_specialist",
            "Looks up quote history and calculates deterministic catalog pricing.",
            self._run_quote_action,
        )
        self.fulfillment_worker = self._create_specialist(
            "fulfillment_specialist",
            "Validates accepted orders and records the corresponding transactions.",
            self._run_fulfillment_action,
        )
        self.advisor_worker = self._create_specialist(
            "business_advisor",
            "Creates internal-only operational recommendations from aggregate results.",
            self._run_advisor_action,
        )
        self.agent = ToolCallingAgent(
            tools=[],
            model=self.workflow_model,
            managed_agents=[
                self.inventory_worker,
                self.quote_worker,
                self.fulfillment_worker,
                self.advisor_worker,
            ],
            name="project_orchestrator",
            description="Delegates each deterministic business action to its specialist.",
            instructions=(
                "Delegate the structured request to the named specialist. Do not make "
                "business decisions or alter its result."
            ),
            max_steps=2,
            verbosity_level=LogLevel.ERROR,
        )
        self.agent.prompt_templates["managed_agent"]["task"] = "{{task}}"
        self.agent.prompt_templates["managed_agent"]["report"] = "{{final_answer}}"

    def _create_specialist(
        self,
        name: str,
        description: str,
        handler: Callable[[str, Dict[str, Any]], Any],
    ) -> ToolCallingAgent:
        agent = ToolCallingAgent(
            tools=[BusinessActionTool(handler)],
            model=self.workflow_model,
            name=name,
            description=description,
            instructions=(
                "Execute the one structured business action provided in the task "
                "using run_business_action, then return its result unchanged."
            ),
            max_steps=2,
            verbosity_level=LogLevel.ERROR,
        )
        agent.prompt_templates["managed_agent"]["task"] = "{{task}}"
        agent.prompt_templates["managed_agent"]["report"] = "{{final_answer}}"
        return agent

    def _run_inventory_action(self, action: str, arguments: Dict[str, Any]) -> Any:
        if action == "assess":
            return self.inventory_agent.assess(
                arguments["requested_items"],
                arguments["request_date"],
                arguments["deadline"],
            )
        if action == "commit_reorders":
            return self.inventory_agent.commit_reorders(arguments["plan"])
        if action == "reserve_order":
            return self.inventory_agent.reserve_order(arguments["plan"])
        raise ValueError(f"Unknown inventory action: {action}.")

    def _run_quote_action(self, action: str, arguments: Dict[str, Any]) -> Any:
        if action != "create_quote":
            raise ValueError(f"Unknown quote action: {action}.")
        return self.quote_agent.create_quote(
            arguments["lines"],
            arguments["customer_brief"],
        )

    def _run_fulfillment_action(self, action: str, arguments: Dict[str, Any]) -> Any:
        if action != "complete":
            raise ValueError(f"Unknown fulfillment action: {action}.")
        return self.order_agent.complete(
            arguments["plan"],
            arguments["quote"],
            arguments["request_date"],
            self.inventory_agent,
        )

    def _run_advisor_action(self, action: str, arguments: Dict[str, Any]) -> Any:
        if action != "analyze":
            raise ValueError(f"Unknown business-advisor action: {action}.")
        return self.business_advisor.analyze(
            arguments["as_of_date"],
            arguments["evaluation_results"],
        )

    def _delegate(
        self,
        agent_name: str,
        action: str,
        arguments: Dict[str, Any],
    ) -> Any:
        task = json.dumps({"action": action, "arguments": arguments})
        result = self.agent.run(
            json.dumps({"agent": agent_name, "task": task})
        )
        if not isinstance(result, str):
            raise TypeError(
                f"Framework delegation to {agent_name} returned "
                f"{type(result).__name__}, expected serialized JSON."
            )
        try:
            return json.loads(result)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Framework delegation to {agent_name} returned invalid JSON."
            ) from exc

    def analyze_business(
        self,
        as_of_date: str,
        evaluation_results: List[Dict],
    ) -> Dict:
        """Delegate internal evaluation results to the internal-only advisor."""
        result = self._delegate(
            "business_advisor",
            "analyze",
            {
                "as_of_date": as_of_date,
                "evaluation_results": evaluation_results,
            },
        )
        if not isinstance(result, dict):
            raise TypeError("Business advisor must return a JSON object.")
        return result

    @staticmethod
    def _prepare_customer_brief(
        request: str,
        request_date: str,
        job: str,
        event: str,
        need_size: str,
    ) -> Dict[str, str]:
        """Translate stated customer context into negotiation inputs, not guesses."""
        return {
            "job": job.strip(),
            "event": event.strip(),
            "need_size": need_size.strip(),
            "deadline": requested_deadline(request, request_date),
            "price_sensitive": str(
                bool(
                    re.search(
                        r"\b(affordable|budget|cheap(?:est)?|discount|lowest price|"
                        r"price-conscious)\b",
                        request,
                        re.IGNORECASE,
                    )
                )
            ).lower(),
        }

    def handle(
        self,
        request: str,
        request_date: str,
        job: str = "",
        event: str = "",
        need_size: str = "",
        progress: Optional[Callable[[str, str], None]] = None,
    ) -> Dict:
        customer_brief = self._prepare_customer_brief(
            request, request_date, job, event, need_size
        )
        if progress:
            context = " / ".join(
                value
                for value in (customer_brief["job"], customer_brief["event"])
                if value
            ) or "customer request"
            progress("Customer advocate", f"prepared {context} requirements")

        parsed_items = parse_requested_items(request)
        if not parsed_items:
            if progress:
                progress("Orchestrator", "request needs item names and quantities")
            return {
                "status": "needs_details",
                "accepted": False,
                "response": "Please specify each requested catalog item and its quantity.",
                "total": 0.0,
                "fulfillment_date": "",
            }

        for item in parsed_items:
            catalog_item, reason = resolve_catalog_item(str(item["description"]))
            if catalog_item is None:
                if progress:
                    progress("Inventory team", f"catalog check: {reason}")
                return {
                    "status": "rejected",
                    "accepted": False,
                    "response": (
                        f"Unable to fulfill this request: {item['description']} — {reason}. "
                        "No inventory or sales transactions were recorded."
                    ),
                    "total": 0.0,
                    "fulfillment_date": "",
                }

        deadline = customer_brief["deadline"]
        plan = self._delegate(
            "inventory_specialist",
            "assess",
            {
                "requested_items": parsed_items,
                "request_date": request_date,
                "deadline": deadline,
            },
        )
        if progress:
            progress(
                "Inventory agent",
                (
                    f"feasible; estimated delivery {plan['fulfillment_date']}"
                    if plan["feasible"]
                    else "; ".join(plan["errors"])
                ),
            )
        if not plan["feasible"]:
            reasons = "; ".join(plan["errors"])
            alternative_dates = re.findall(
                r"earliest estimated date (\d{4}-\d{2}-\d{2})",
                reasons,
            )
            if alternative_dates:
                alternative_deadline = max(alternative_dates)
                alternative_plan = self._delegate(
                    "inventory_specialist",
                    "assess",
                    {
                        "requested_items": parsed_items,
                        "request_date": request_date,
                        "deadline": alternative_deadline,
                    },
                )
                if alternative_plan["feasible"]:
                    alternative_quote = self._delegate(
                        "quote_specialist",
                        "create_quote",
                        {
                            "lines": alternative_plan["lines"],
                            "customer_brief": customer_brief,
                        },
                    )
                    if progress:
                        progress(
                            "Customer advocate",
                            f"negotiated an alternate delivery offer for {alternative_deadline}",
                        )
                        progress(
                            "Quote agent",
                            f"${alternative_quote['total']:,.2f} total at the alternate date",
                        )
                    alternate_items = ", ".join(
                        f"{line['quantity']:,} {line['item_name']}"
                        for line in alternative_plan["lines"]
                    )
                    return {
                        "status": "quoted_alternative_pending_approval",
                        "accepted": False,
                        "response": (
                            f"We cannot meet the requested delivery date of {deadline}: "
                            f"{reasons}. For your {event or 'order'} request, we can offer "
                            f"{alternate_items} for ${alternative_quote['total']:,.2f} "
                            f"with estimated delivery by {alternative_deadline}. "
                            f"{alternative_quote['price_rationale']} "
                            "This is an alternative offer, not an order; reply to approve "
                            "the later date."
                        ),
                        "total": alternative_quote["total"],
                        "fulfillment_date": alternative_deadline,
                    }
            return {
                "status": "rejected",
                "accepted": False,
                "response": (
                    f"Unable to meet the requested delivery date of {deadline}: {reasons}. "
                    "No transactions were recorded."
                ),
                "total": 0.0,
                "fulfillment_date": "",
            }

        quote = self._delegate(
            "quote_specialist",
            "create_quote",
            {
                "lines": plan["lines"],
                "customer_brief": customer_brief,
            },
        )
        if progress:
            progress(
                "Quote agent",
                f"${quote['total']:,.2f} total; {quote['price_rationale']}",
            )
        item_summary = ", ".join(
            f"{line['quantity']:,} {line['item_name']}" for line in plan["lines"]
        )
        response = (
            f"Quote for {item_summary}: subtotal ${quote['subtotal']:,.2f}; "
            f"bulk discount {quote['discount_rate']:.0%} "
            f"(-${quote['discount']:,.2f}); total ${quote['total']:,.2f}. "
            f"{quote['price_rationale']} {quote['history_note']} "
            f"Estimated delivery: {plan['fulfillment_date']}."
        )
        has_purchase_intent = bool(
            re.search(r"\b(order|purchase|buy|confirm the order)\b", request, re.IGNORECASE)
        )
        if not has_purchase_intent:
            if progress:
                progress("Customer advocate", "quote ready; awaiting customer approval")
            return {
                "status": "quoted_pending_approval",
                "accepted": False,
                "response": response + " Reply with acceptance to place the order.",
                "total": quote["total"],
                "fulfillment_date": plan["fulfillment_date"],
            }

        plan["request_date"] = request_date
        line_count = self._delegate(
            "fulfillment_specialist",
            "complete",
            {
                "plan": plan,
                "quote": quote,
                "request_date": request_date,
            },
        )
        if progress:
            progress(
                "Fulfillment agent",
                f"recorded {line_count} line items for {plan['fulfillment_date']}",
            )
        return {
            "status": "fulfilled",
            "accepted": True,
            "response": (
                response
                + f" Order accepted and recorded for delivery by {plan['fulfillment_date']} "
                f"({line_count} line items)."
            ),
            "total": quote["total"],
            "fulfillment_date": plan["fulfillment_date"],
        }


def run_test_scenarios():
    """Evaluate the system on the supplied sample requests and save a CSV report."""
    print("Initializing database...")
    init_database()
    try:
        requests_df = pd.read_csv(PROJECT_DIR / "quote_requests_sample.csv")
        parsed_dates = []
        for raw_date in requests_df["request_date"]:
            try:
                parsed_dates.append(
                    datetime.strptime(str(raw_date), "%m/%d/%y").strftime("%Y-%m-%d")
                )
            except ValueError:
                parsed_dates.append(None)
        requests_df.loc[:, "request_date"] = parsed_dates
        requests_df.dropna(subset=["request_date"], inplace=True)
        requests_df = requests_df.sort_values("request_date")
    except Exception as exc:
        raise RuntimeError(f"Could not load the supplied test requests: {exc}") from exc

    first_date = str(requests_df["request_date"].min())
    initial_report = financial_report_tool.forward(as_of_date=first_date)
    current_cash = float(initial_report["cash_balance"])
    current_inventory = float(initial_report["inventory_value"])
    orchestrator = ProjectOrchestrator()
    results = []

    for request_number, (_, row) in enumerate(requests_df.iterrows(), start=1):
        request_date = str(row["request_date"])
        progress_display = TerminalProgressDisplay(enabled=sys.stdout.isatty())
        print(f"\n=== Request {request_number} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        outcome = orchestrator.handle(
            request=str(row["request"]),
            request_date=request_date,
            job=str(row["job"]),
            event=str(row["event"]),
            need_size=str(row.get("need_size", "")),
            progress=progress_display.update,
        )
        response = str(outcome["response"])

        report = financial_report_tool.forward(as_of_date=request_date)
        current_cash = float(report["cash_balance"])
        current_inventory = float(report["inventory_value"])
        print(f"Status: {outcome['status']}")
        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": request_number,
                "request_date": request_date,
                "job": row["job"],
                "event": row["event"],
                "status": outcome["status"],
                "accepted": outcome["accepted"],
                "fulfillment_date": outcome["fulfillment_date"],
                "quoted_total": outcome["total"],
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

    last_date = str(requests_df["request_date"].max())
    final_report = financial_report_tool.forward(as_of_date=last_date)
    fulfilled = sum(bool(result["accepted"]) for result in results)
    rejected = sum(result["status"] == "rejected" for result in results)
    advisor_report = orchestrator.analyze_business(last_date, results)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")
    print(f"Orders fulfilled: {fulfilled}; requests rejected: {rejected}")

    print("\n===== INTERNAL BUSINESS ADVISOR =====")
    for recommendation in advisor_report["recommendations"]:
        print(f"- {recommendation}")
    advisor_path = PROJECT_DIR / "business_advisor_report.txt"
    advisor_lines = [
        f"Business advisor analysis as of {last_date}",
        f"Accepted sample requests: {advisor_report['accepted_requests']}",
        f"Rejected sample requests: {advisor_report['rejected_requests']}",
        "",
        "Transaction totals:",
    ]
    advisor_lines.extend(
        (
            f"- {item['transaction_type']}: {item['transaction_count']} "
            f"transactions, {item['units']:,} units, ${item['amount']:,.2f}"
        )
        for item in advisor_report["transaction_totals"]
    )
    advisor_lines.extend(["", "Recommendations:"])
    advisor_lines.extend(f"- {item}" for item in advisor_report["recommendations"])
    advisor_path.write_text("\n".join(advisor_lines) + "\n", encoding="utf-8")

    results_path = PROJECT_DIR / "test_results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Evaluation saved to {results_path}")
    print(f"Advisor report saved to {advisor_path}")
    return results

if __name__ == "__main__":
    run_test_scenarios()
