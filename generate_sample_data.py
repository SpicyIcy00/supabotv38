#!/usr/bin/env python3
"""
Generate sample data for testing the mobile dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data():
    """Generate sample data for testing"""
    
    st.title("📊 Generate Sample Data")
    st.markdown("This will populate your database with sample data for testing the mobile dashboard")
    
    if st.button("🚀 Generate Sample Data"):
        try:
            from supabot.core.database import get_db_manager
            db_manager = get_db_manager()
            conn = db_manager.create_connection()
            
            if not conn:
                st.error("❌ Cannot connect to database")
                return
            
            cur = conn.cursor()
            
            # Check if data already exists
            cur.execute("SELECT COUNT(*) FROM transactions")
            transaction_count = cur.fetchone()[0]
            
            if transaction_count > 0:
                st.warning(f"⚠️ Database already has {transaction_count} transactions. Do you want to add more sample data?")
                if not st.button("Yes, add more data"):
                    return
            
            with st.spinner("Generating sample data..."):
                
                # Generate stores
                stores = [
                    ("Rockwell", "Makati"),
                    ("Greenhills", "San Juan"),
                    ("Magnolia", "Quezon City"),
                    ("North Edsa", "Quezon City"),
                    ("Fairview", "Quezon City")
                ]
                
                for store_name, location in stores:
                    cur.execute(
                        "INSERT INTO stores (name, location) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
                        (store_name, location)
                    )
                
                # Generate products
                products = [
                    ("iPhone 15 Pro", "Smartphones", 89990),
                    ("Samsung Galaxy S24", "Smartphones", 79990),
                    ("MacBook Pro M3", "Laptops", 129990),
                    ("iPad Air", "Tablets", 49990),
                    ("AirPods Pro", "Accessories", 24990),
                    ("Apple Watch Series 9", "Wearables", 39990),
                    ("Sony WH-1000XM5", "Accessories", 29990),
                    ("Nintendo Switch OLED", "Gaming", 19990),
                    ("DJI Mini 3 Pro", "Drones", 59990),
                    ("GoPro Hero 12", "Cameras", 34990)
                ]
                
                for product_name, category, price in products:
                    cur.execute(
                        "INSERT INTO products (product_name, category, price) VALUES (%s, %s, %s) ON CONFLICT (product_name) DO NOTHING",
                        (product_name, category, price)
                    )
                
                # Get store and product IDs
                cur.execute("SELECT id FROM stores")
                store_ids = [row[0] for row in cur.fetchall()]
                
                cur.execute("SELECT id, price FROM products")
                product_data = cur.fetchall()
                product_ids = [row[0] for row in product_data]
                product_prices = {row[0]: row[1] for row in product_data}
                
                # Generate transactions for the last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                transactions = []
                transaction_id = 1
                
                for i in range(1000):  # Generate 1000 transactions
                    # Random date within last 30 days
                    transaction_date = start_date + timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(9, 21),
                        minutes=random.randint(0, 59)
                    )
                    
                    # Random store and product
                    store_id = random.choice(store_ids)
                    product_id = random.choice(product_ids)
                    quantity = random.randint(1, 3)
                    price = product_prices[product_id]
                    total = price * quantity
                    
                    transactions.append((
                        transaction_id,
                        store_id,
                        product_id,
                        quantity,
                        price,
                        total,
                        transaction_date,
                        'sale'
                    ))
                    transaction_id += 1
                
                # Insert transactions in batches
                batch_size = 100
                for i in range(0, len(transactions), batch_size):
                    batch = transactions[i:i + batch_size]
                    cur.executemany(
                        """
                        INSERT INTO transactions 
                        (id, store_id, product_id, quantity, price, total, transaction_date, transaction_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        batch
                    )
                
                conn.commit()
                
                # Verify data
                cur.execute("SELECT COUNT(*) FROM stores")
                store_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM products")
                product_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM transactions")
                transaction_count = cur.fetchone()[0]
                
                st.success(f"✅ Sample data generated successfully!")
                st.info(f"""
                📊 **Generated Data:**
                - **Stores**: {store_count}
                - **Products**: {product_count}
                - **Transactions**: {transaction_count}
                """)
                
                st.markdown("### 🎯 Next Steps:")
                st.markdown("""
                1. **Test the Dashboard**: Go back to the main dashboard
                2. **Try Different Filters**: Test with '7D', '1M', and different stores
                3. **Mobile Testing**: Use the mobile toggle in the sidebar
                """)
                
                conn.close()
                
        except Exception as e:
            st.error(f"❌ Error generating sample data: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

def main():
    st.set_page_config(
        page_title="Sample Data Generator",
        page_icon="📊",
        layout="wide"
    )
    
    generate_sample_data()

if __name__ == "__main__":
    main()
