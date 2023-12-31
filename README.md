# RelNetCare
RelNetCare (Relation Network for Elderly Health Care) is a project developed as part of my Master Thesis at the Technical University of Munich ([TUM](https://www.tum.de/)) under the supervision of the [sebis](https://wwwmatthes.in.tum.de/pages/t5ma0jrv6q7k/sebis-Public-Website-Home) chair. 

## Project's Goal
The project aims to develop a chatbot that utilizes an automatically generated relation network (Personal Knowledge Graph) based on the chat history of elderly users. The chatbot provides personalized topic recommendations to enhance their engagement and contribute to a healthier and happier life.

## Getting Started

### Environment Variables
The following environment variables are necessary for this project:

* `BERT_BASE_DIR`: Base directory path of the BERT model.
* `NEO4J_USERNAME`: Username for Neo4j database connection.
* `NEO4J_PASSWORD`: Password for Neo4j database connection.

### Installation

1. Create and activate environment
> Requirements: `Python 3.10` and `pipenv`


```bash
pipenv shell
```

2. Install dependencies 
    
```bash
pip install -r requirements.txt
```

3. Confirm Weights & Biases installation. More info: [click here](https://wandb.ai/quickstart/pytorch).

```bash
pip install wandb
wandb login
```

4. Install Neo4j. More info: [click here](https://neo4j.com/).

4. Enjoy exploring RelNetCare! 😉