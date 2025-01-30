# Kedro Pokemon Legendary

## Kedro-Pipelines: Is Pokèmon Legendary?

This repository demonstrates how Kedro pipelines can be applied to a classification problem to determine whether a Pokémon is Legendary based on its characteristics. 

### About the dataset

[Source and credits](https://www.kaggle.com/datasets/abcsds/pokemon?resource=download)

The dataset used in this project contains information about 721 Pokémon, including their attributes and stats. 
Each Pokémon is described by the following features:

- **#**: ID for each pokemon
- **Name**: Name of each pokemon
- **Type 1**: Each pokemon has a type, this determines weakness/resistance to attacks
- **Type 2**: Some pokemon are dual type and have 2
- **Total**: sum of all stats that come after this, a general guide to how strong a pokemon is
- **HP**: hit points, or health, defines how much damage a pokemon can withstand before fainting
- **Attack**: the base modifier for normal attacks (eg. Scratch, Punch)
- **Defense**: the base damage resistance against normal attacks
- **SP Atk**: special attack, the base modifier for special attacks (e.g. fire blast, bubble beam)
- **SP Def**: the base damage resistance against special attacks
- **Speed**: determines which pokemon attacks first each round

#### Source and credits
The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/abcsds/pokemon?resource=download) and is credited to Myles O'Neill.

**Requirements**
To run these notebooks locally, ensure you have Python installed along with the following libraries:

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter notebook

### **Usage**

1. Clone this repository to your local machine:

git clone https://github.com/DoctorOzymandias95/kedro_pokemon_legendary.git