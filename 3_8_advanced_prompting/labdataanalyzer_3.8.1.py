"""
Практический пример: Анализ лабораторных данных с визуализацией
Демонстрация цепочки: Структурирование → Анализ → Визуализация
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import requests
from decouple import config

class LabDataAnalyzer:
    """Анализ лабораторных данных с визуализацией"""
    
    def __init__(self, llm_api="ollama"):
        self.llm_api = llm_api
        
    def generate_sample_data(self):
        """Генерация демо-данных эксперимента по синтезу наночастиц"""
        print("📊 Генерация демонстрационных данных...")
        
        # Сырые данные в разных форматах (как бывают в лабораторных журналах)
        raw_data = """
        Эксперимент №45: Синтез наночастиц золота
        Дата: 15.11.2024
        Лаборант: Иванов А.И.
        
        Серия 1: Влияние концентрации восстановителя
        -------------------------------------------
        № пробы | Конц. NaBH4 (мМ) | Время (мин) | Размер (нм) | Выход (%)
        1       | 10               | 30          | 15.2        | 45
        2       | 20               | 30          | 12.8        | 62
        3       | 30               | 30          | 10.5        | 78
        4       | 40               | 30          | 8.3         | 85
        5       | 50               | 30          | 7.1         | 88
        
        Заметки: При концентрации выше 50 мМ наблюдается агрегация частиц.
        Оптимальный pH = 7.2, температура 25°C.
        Использован HAuCl4 * 3H2O в качестве прекурсора.
        """
        return raw_data
    
    def query_llm(self, prompt, conversation_history=None):
        """Запрос к LLM"""
        if self.llm_api == "ollama":
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen3:0.6b", "prompt": prompt, "stream": False}
            )
            return response.json().get("response", "")
        else:
            OPENROUTER_KEY = config('OPENROUTER_API_KEY')
            OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
            OPENROUTER_MODEL = "qwen/qwen3-coder-next"

            """Запрос к OpenRouter с поддержкой контекста"""
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json"
                }
                
                # Формируем историю сообщений
                messages = []
                if conversation_history:
                    for i, msg in enumerate(conversation_history[-5:]):  # Последние 5
                        role = "user" if i % 2 == 0 else "assistant"
                        messages.append({"role": role, "content": msg})
                
                messages.append({"role": "user", "content": prompt})
                
                data = {
                    "model": OPENROUTER_MODEL,
                    "messages": messages,
                    "temperature": 0.3
                }
                
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    print(response.json()["choices"][0]["message"]["content"])
                    print(response.status_code)
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print(response.status_code)
                return ""
            except:
                print(response.status_code)
                return ""
    
    def structure_raw_data(self, raw_data):
        """Шаг 1: Структурирование сырых данных"""
        print("\n🔧 Шаг 1: Структурирование сырых данных...")
        
        prompt = f"""
        Преобразуй лабораторные данные в структурированный JSON.
        
        Сырые данные:
        {raw_data}
        
        Извлеки:
        1. Название эксперимента
        2. Переменные исследования (независимые и зависимые)
        3. Данные в виде таблицы
        4. Условия эксперимента
        5. Заметки и наблюдения
        
        Верни ТОЛЬКО JSON в формате:
        {{
            "experiment_name": "название",
            "variables": {{"independent": ["переменная1"], "dependent": ["переменная2", "переменная3(если есть)"]}},
            "data": [
                {{"sample": 1, "concentration": 10, "size": 15.2, "yield": 45}},
                ...
            ],
            "conditions": "условия",
            "notes": "заметки"
        }}
        """
        
        response = self.query_llm(prompt)
        
        # Извлекаем JSON из ответа
        try:
            # Ищем JSON в ответе
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            structured = json.loads(json_str)
            print("✓ Данные структурированы")
            return structured
        except:
            print("✗ Ошибка парсинга JSON, используем демо-данные")
            # Возвращаем структурированные демо-данные
            return {
                "experiment_name": "Синтез наночастиц золота",
                "variables": {
                    "independent": ["Концентрация NaBH4 (мМ)"],
                    "dependent": ["Размер частиц (нм)", "Выход (%)"]
                },
                "data": [
                    {"sample": 1, "concentration": 10, "size": 15.2, "yield": 45},
                    {"sample": 2, "concentration": 20, "size": 12.8, "yield": 62},
                    {"sample": 3, "concentration": 30, "size": 10.5, "yield": 78},
                    {"sample": 4, "concentration": 40, "size": 8.3, "yield": 85},
                    {"sample": 5, "concentration": 50, "size": 7.1, "yield": 88}
                ],
                "conditions": "pH = 7.2, температура 25°C",
                "notes": "При концентрации выше 50 мМ наблюдается агрегация частиц"
            }
    
    def analyze_data(self, structured_data):
        """Шаг 2: Статистический анализ"""
        print("\n📈 Шаг 2: Статистический анализ...")
        
        # Извлекаем данные для анализа
        concentrations = [d["concentration"] for d in structured_data["data"]]
        sizes = [d["size"] for d in structured_data["data"]]
        yields = [d["yield"] for d in structured_data["data"]]
        
        # Рассчитываем статистику
        size_mean = np.mean(sizes)
        size_std = np.std(sizes)
        yield_mean = np.mean(yields)
        yield_std = np.std(yields)
        
        # Линейная регрессия для размера vs концентрация
        slope_size, intercept_size, r_value_size, p_value_size, std_err_size = stats.linregress(
            concentrations, sizes
        )
        
        # Линейная регрессия для выхода vs концентрация
        slope_yield, intercept_yield, r_value_yield, p_value_yield, std_err_yield = stats.linregress(
            concentrations, yields
        )
        
        # 95% доверительные интервалы
        n = len(sizes)
        t_value = stats.t.ppf(0.95, n-1)  # t-критерий для 95% ДИ
        
        size_ci = t_value * size_std / np.sqrt(n)
        yield_ci = t_value * yield_std / np.sqrt(n)
        
        analysis_results = {
            "size_stats": {
                "mean": round(size_mean, 2),
                "std": round(size_std, 2),
                "ci_95": round(size_ci, 2),
                "regression": {
                    "slope": round(slope_size, 3),
                    "intercept": round(intercept_size, 3),
                    "r_squared": round(r_value_size**2, 3),
                    "p_value": round(p_value_size, 4)
                }
            },
            "yield_stats": {
                "mean": round(yield_mean, 1),
                "std": round(yield_std, 1),
                "ci_95": round(yield_ci, 1),
                "regression": {
                    "slope": round(slope_yield, 3),
                    "intercept": round(intercept_yield, 3),
                    "r_squared": round(r_value_yield**2, 3),
                    "p_value": round(p_value_yield, 4)
                }
            }
        }
        
        # Интерпретация через LLM
        prompt = f"""
        Интерпретируй результаты статистического анализа:
        
        Эксперимент: {structured_data['experiment_name']}
        
        Статистика размера частиц:
        - Средний размер: {size_mean:.1f} ± {size_std:.1f} нм
        - 95% доверительный интервал: ±{size_ci:.1f} нм
        - Зависимость от концентрации: размер = {slope_size:.3f} * конц. + {intercept_size:.1f}
        - R² = {r_value_size**2:.3f}, p-value = {p_value_size:.4f}
        
        Статистика выхода:
        - Средний выход: {yield_mean:.1f} ± {yield_std:.1f} %
        - 95% доверительный интервал: ±{yield_ci:.1f} %
        - Зависимость от концентрации: выход = {slope_yield:.3f} * конц. + {intercept_yield:.1f}
        - R² = {r_value_yield**2:.3f}, p-value = {p_value_yield:.4f}
        
        Ответь на вопросы:
        1. Какая тенденция наблюдается?
        2. Статистически значимы ли зависимости?
        3. Какая концентрация оптимальна?
        4. Какие дальнейшие эксперименты предложить?
        """
        
        interpretation = self.query_llm(prompt)
        
        return analysis_results, interpretation
    
    def visualize_results(self, structured_data, analysis_results):
        """Шаг 3: Визуализация результатов"""
        print("\n🎨 Шаг 3: Создание визуализаций...")
        
        # Извлекаем данные
        concentrations = [d["concentration"] for d in structured_data["data"]]
        sizes = [d["size"] for d in structured_data["data"]]
        yields = [d["yield"] for d in structured_data["data"]]
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # График 1: Размер частиц
        ax1.plot(concentrations, sizes, 'bo-', linewidth=2, markersize=8, label='Экспериментальные данные')
        
        # Линия регрессии
        x_fit = np.linspace(min(concentrations), max(concentrations), 100)
        slope = analysis_results["size_stats"]["regression"]["slope"]
        intercept = analysis_results["size_stats"]["regression"]["intercept"]
        y_fit = slope * x_fit + intercept
        ax1.plot(x_fit, y_fit, 'r--', alpha=0.7, label=f'Линейная регрессия (R²={analysis_results["size_stats"]["regression"]["r_squared"]})')
        
        ax1.set_xlabel('Концентрация NaBH4 (мМ)', fontsize=12)
        ax1.set_ylabel('Размер частиц (нм)', fontsize=12)
        ax1.set_title('Зависимость размера частиц от концентрации', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Добавляем аннотацию со статистикой
        stats_text = f"Средний размер: {analysis_results['size_stats']['mean']} ± {analysis_results['size_stats']['std']} нм\n"
        stats_text += f"95% ДИ: ±{analysis_results['size_stats']['ci_95']} нм\n"
        stats_text += f"p-value: {analysis_results['size_stats']['regression']['p_value']}"
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # График 2: Выход продукта
        bars = ax2.bar(range(len(yields)), yields, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Номер пробы', fontsize=12)
        ax2.set_ylabel('Выход (%)', fontsize=12)
        ax2.set_title('Выход продукта по пробам', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(yields)))
        ax2.set_xticklabels([f"Проба {i+1}" for i in range(len(yields))])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, yield_val, conc in zip(bars, yields, concentrations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{yield_val}%\n({conc} мМ)',
                    ha='center', va='bottom', fontsize=10)
        
        # Линия среднего значения с доверительным интервалом
        mean_yield = analysis_results["yield_stats"]["mean"]
        ci = analysis_results["yield_stats"]["ci_95"]
        ax2.axhline(y=mean_yield, color='red', linestyle='--', alpha=0.7, label=f'Среднее: {mean_yield:.1f}%')
        ax2.fill_between([-0.5, len(yields)-0.5], 
                        mean_yield - ci, mean_yield + ci, 
                        color='red', alpha=0.1, label=f'95% ДИ: ±{ci:.1f}%')
        ax2.legend()
        
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig('lab_analysis_results.png', dpi=300, bbox_inches='tight')
        print("✓ График сохранен как 'lab_analysis_results.png'")
        
        plt.show()
        
        return fig
    
    def run_full_analysis(self):
        """Полный цикл анализа"""
        print("="*60)
        print("АНАЛИЗ ЛАБОРАТОРНЫХ ДАННЫХ: ПОЛНЫЙ ЦИКЛ")
        print("="*60)
        
        # Шаг 0: Генерация демо-данных
        raw_data = self.generate_sample_data()
        print(f"\n📄 Сырые данные:\n{raw_data[:200]}...")
        
        # Шаг 1: Структурирование
        structured_data = self.structure_raw_data(raw_data)
        print(f"\n📋 Структурированные данные:")
        print(json.dumps(structured_data, indent=2, ensure_ascii=False)[:300] + "...")
        
        # Шаг 2: Анализ
        analysis_results, interpretation = self.analyze_data(structured_data)
        print(f"\n📊 Результаты анализа:")
        print(json.dumps(analysis_results, indent=2, ensure_ascii=False))
        print(f"\n🤖 Интерпретация:\n{interpretation}")
        
        # Шаг 3: Визуализация
        fig = self.visualize_results(structured_data, analysis_results)
        
        # Шаг 4: Генерация отчета
        self.generate_report(structured_data, analysis_results, interpretation)
        
        return {
            "structured_data": structured_data,
            "analysis": analysis_results,
            "interpretation": interpretation
        }
    
    def generate_report(self, structured_data, analysis, interpretation):
        """Генерация итогового отчета"""
        print("\n📝 Шаг 4: Генерация отчета...")
        
        prompt = f"""
        Напиши краткий отчет об эксперименте в markdown. ГОСТ, 14 пт, 1,5 интервал
        
        Данные эксперимента: {json.dumps(structured_data, ensure_ascii=False)}
        Результаты анализа: {json.dumps(analysis, ensure_ascii=False)}
        Интерпретация: {interpretation}
        
        Структура отчета:
        1. Цель эксперимента
        2. Методика
        3. Результаты
        4. Выводы
        5. Рекомендации
        
        Объем: 300-400 слов.
        """
        
        report = self.query_llm(prompt)
        
        with open('experiment_report.md', 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО ЭКСПЕРИМЕНТУ\n")
            f.write("="*50 + "\n\n")
            f.write(report)
        
        print("✓ Отчет сохранен как 'experiment_report.md'")
        return report

# Запуск демонстрации
if __name__ == "__main__":
    analyzer = LabDataAnalyzer(llm_api="openrouter")
    results = analyzer.run_full_analysis()