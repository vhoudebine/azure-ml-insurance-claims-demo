## How to setup Promptflow for local Development

__**Note: AI Search Promptflow Connection is still in Preview and is not yet supported for local development**__

### Step 0
- Setup your Azure Open AI endpoint via the Azure portal
- Setup your Azure AI Search index with the [pdf policy data](../data/policy_pdfs/) via the Azure portal

### Step 1
- Ensure PromptFlow python sdk is installed  
```pip install promptflow```

### Step 2
- Activate Azure Open AI and AI Search connections
  - Change variables in the yaml files in the [connections](./connections/) folder to include your specific variables
- Establish the connections  
  - For Open AI, run ```pf connection create --file ./promptflow/connections/<FILENAME>.yaml --set api_key={api_key} api_base={api_base} --name {connection_name}``` from the terminal  

### Step 3
- Ensure conneciton variables in your [flow](./policy_flow/flow.dag.yaml) match the connections you set up