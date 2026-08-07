import pandas as pd

def create_report_table():
    # Sample data for the report
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Department': ['HR', 'Finance', 'IT', 'Marketing']
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    df.to_csv('report_table.csv', index=False)  # Save the report table to a CSV file

if __name__ == "__main__":
    create_report_table()
    print("Report table created and saved as 'report_table.csv'.")