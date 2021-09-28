# oneNeuron
oneNeuron | Perceptron


# commands used -

```bash
git add . && git commit -m " docstrings updated" && git push origin main
```

```bash
cp Research\ notebooks/demo.ipynb .
```

 <a href="https://www.w3schools.com">Visit W3Schools.com!</a> 


## Add URL -
[GitHub Handbook](https://guides.github.com/introduction/git-handbook/)

## Add image -
![Sample Image](plots/and.png)

 <img src="plots/and.png" alt="Girl in a jacket" width="500" height="600"> 


 ## Python code
    
 ```python   
 def main(data, eta, epochs, filename, plotFileName):
    df = pd.DataFrame(data)
    print(df)
    X,y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss() # dummy variable
    save_model(model, filename=filename)
    save_plot(df, plotFileName, model)

 ```