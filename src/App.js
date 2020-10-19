import React from 'react';
import './App.css';
import axios from 'axios'

class App extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      algorithm: null,
      file : null,
      predict: false,
      accuracy: null
    }
  }

  changeHandler = (e)=>{
    var tmppath = URL.createObjectURL(e.target.files[0]);
    document.getElementsByTagName('img')[0].setAttribute('src',URL.createObjectURL(e.target.files[0]))
    this.setState({...this.state,
      file : e.target.files[0]
    })
  }

  clickHandler = ()=>{
    
    if ((this.state.file !== null) && (this.state.algorithm !== null)){
      var formData = new FormData();
      formData.append("fileToUpload", this.state.file, this.state.file.name);
      formData.append("algorithm", this.state.algorithm)
      console.log('This is file, ', this.state.file.path)

      axios.post('http://192.168.1.12:5000/image', formData)
      .then((res)=>{
        console.log('JSON data', res.data)
        this.setState({
          ...this.state,
          predict : res.data['value'],
          accuracy : res.data['accuracy'] 
        })
        //console.log(res.data)
      })
      .catch((err)=>{
        console.log('Error generated, ', err)
      })
    }
    else{
      alert('Please Make Sure you have selected an algorithm and inserted image for prediction.')
    }   
  }

  selectChange = (e)=>{
    this.setState({
      ...this.state,
      algorithm : e.target.value
    },()=>{
      console.log(this.state)
    })
    
  }

  render(){
    return(
      <div className="App">
      <h2>Hand Written Digits from 0-9 recognition</h2>
      <form method='POST' onSubmit={e=>{ e.preventDefault()}} className="form-class">
        File <input type="file" id="visitorphoto" name="visitorPhoto" accept="image/*" onChange={this.changeHandler}/>
        <br />
        <h4>Select an algorithm</h4>
        <select id="cars" onChange={this.selectChange}>
          <option label="Linear Regression">Linear Regression</option>
          <option label="Logistic Regression">Logistic Regression</option>
          <option label="Support vector classification">Support vector classification</option>
          <option label="Support vector Regression">Support vector Regression</option>
          <option label="Stochastic Gradient Descent">Stochastic Gradient Descent</option>
          <option label="K-Nearest Neighbours">K-Nearest Neighbours</option>
        </select>
        <br />
        <br />
        <button onClick={this.clickHandler}>Send image for prediction</button>
      </form>
      <br />
      <img src="" width={200} height={200} alt="Image for prediction"/>
      {
        this.state.predict? <><h1 style={{paddingBottom:0, marginBottom: 0}}>Predicted value: {Math.round(this.state.predict)}</h1> <h3>Modal Used : {this.state.algorithm}</h3> <h3 style={{paddingBottom:10}}>Accuracy of the modal: {parseFloat(this.state.accuracy)*100}</h3></>: null 
      }
    </div>
    )
  }
   
}

export default App;
