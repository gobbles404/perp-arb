flowchart TB
subgraph Data["Data Layer"]
CSV["CSV Files"] --> Loader["Data Loader"]
Loader --> MarketData["Market Data Objects"]
end

    subgraph Engine["Backtesting Engine"]
        MarketData --> EventQueue["Event Queue"]
        EventQueue --> Strategies["Strategy Modules"]
        Strategies --> SignalGen["Signal Generation"]
        SignalGen --> RiskMgr["Risk Manager"]
        RiskMgr --> OrderGen["Order Generation"]
        OrderGen --> Broker["Broker (Execution)"]
        Broker --> Portfolio["Portfolio"]
        Portfolio --> PnL["PnL Calculation"]
    end

    subgraph Analysis["Analysis Layer"]
        PnL --> Metrics["Performance Metrics"]
        Metrics --> Reporting["Report Generation"]
        Reporting --> Visuals["Visualization (PNG)"]
        Reporting --> DataExport["Data Export (CSV/JSON)"]
    end

    classDef data fill:#e6f3ff,stroke:#3385ff
    classDef engine fill:#fff2e6,stroke:#ff9933
    classDef analysis fill:#e6ffe6,stroke:#33cc33

    class Data,CSV,Loader,MarketData data
    class Engine,EventQueue,Strategies,SignalGen,RiskMgr,OrderGen,Broker,Portfolio,PnL engine
    class Analysis,Metrics,Reporting,Visuals,DataExport analysis
