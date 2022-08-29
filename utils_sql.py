import logging
import pandas as pd
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 10000)
pd.set_option("display.max_colwidth", 10000)

# A logger for this file
log = logging.getLogger(__name__)


class SQLqueriesDB:
    """
    Queries for SQL DB
    """
    def __init__(self, con, db_name):

        self.con = con
        self.db_name = db_name

    def get_all_table(self, table_name):

        table_df = pd.read_sql_query("SELECT * "
                                   " FROM {}.{}".format(self.db_name, table_name), self.con)
        log.info("table {} shape: {}".format(table_name, table_df.shape))
        return table_df

    def get_distinct_col(self, table_name, col_name):

        table_df = pd.read_sql_query("SELECT DISTINCT {} "
                                   " FROM {}.{}".format(col_name, self.db_name, table_name), self.con)
        log.info("Distinct col {} in table {} shape: {}".format(col_name, table_name, table_df.shape))
        return table_df

    def get_all_table_where_col_in_list(self, table_name, col_name, list):

        table_df = pd.read_sql_query("SELECT * "
                                   " FROM {}.{} "
                                   " WHERE {} in {}".format(self.db_name, table_name, col_name, tuple(list)), self.con)
        log.info("table {} shape: {}".format(table_name, table_df.shape))
        return table_df

    def get_all_table_where_col_in_sub_query(self, table_name, col_name, sub_query):

        table_df = pd.read_sql_query("SELECT * "
                                   " FROM {}.{} "
                                   " WHERE {} in ({})".format(self.db_name, table_name, col_name, sub_query), self.con)
        log.info("table {} shape: {}".format(table_name, table_df.shape))
        return table_df

    def get_distinct_col_where_col_in_list(self, table_name, col_name, where_col_name, list):

        table_df = pd.read_sql_query("SELECT DISTINCT {} "
                                   " FROM {}.{} "
                                   " WHERE {} in {}".format(col_name, self.db_name, table_name,  where_col_name,
                                                            tuple(list)), self.con)
        log.info("Distinct col {} in table {} shape: {}".format(col_name, table_name, table_df.shape))
        return table_df

    def get_col_where_col_in_list(self, table_name, col_name, where_col_name, list):

        table_df = pd.read_sql_query("SELECT {} "
                                   " FROM {}.{} "
                                   " WHERE {} in {}".format(col_name, self.db_name, table_name,  where_col_name,
                                                            tuple(list)), self.con)
        log.info(" col {} in table {} shape: {}".format(col_name, table_name, table_df.shape))
        return table_df

    def get_custom_query(self, query):
        table_df = pd.read_sql_query(query, self.con)
        log.info(" custom query shape: {}".format(table_df.shape))
        return table_df




