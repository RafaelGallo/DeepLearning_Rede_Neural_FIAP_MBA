from setuptools import setup, find_packages

setup(
    name="quantumfinance",
    version="0.1.0",
    description="Projeto Deep Learning e ETL para previsão de ações com múltiplas arquiteturas.",
    author="Seu Nome",
    author_email="seu@email.com",
    packages=find_packages(),  # Busca automaticamente todos os pacotes python (com __init__.py)
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tensorflow>=2.9",  # ou keras, se usar só keras
        # adicione outros pacotes se necessário
    ],
    include_package_data=True,
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "quantumfinance-main=main:main"
        ]
    },
    # Se quiser incluir arquivos de dados, pode usar:
    # package_data={"": ["data/*.csv"]}
)
