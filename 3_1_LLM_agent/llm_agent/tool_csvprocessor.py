import pandas as pd


class CSVProcessorTool:

    name = "csvprocessor"
    description = (
        "Читает CSV-файл, фильтрует его строки ."
        "И Возвращает статистику по данным."
    )

    def use(self, file_path, operation="read", col=None, val= None ):
        df = pd.read_csv(file_path)

        if operation == "read":
            return df.head().to_string(index=False)
        
        
        elif operation == "filter":
            filtered = df[df[col] == val]
            return filtered.to_string(index=False)
        
        elif operation == "stats":
            stats = df[col].describe()
            return stats.to_string()



