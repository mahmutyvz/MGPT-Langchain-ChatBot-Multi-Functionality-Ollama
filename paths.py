class Path:
    """
    This class encapsulates paths and constants related to file locations and dataset properties.

    Attributes:

    sql_path: The MSSQL driver connection path.
    pdf_save_path: The pdf path that the chatbot uses to analyze
    """
    sql_path = 'mssql+pyodbc://DESKTOP-GU7QGA2\\MAHMUTYAVUZ/etrade?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
    pdf_save_path = 'pdf_chatbot'