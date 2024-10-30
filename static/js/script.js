document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const vaccineType = document.getElementById('vaccine_type').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            age: parseInt(age), 
            gender: parseInt(gender), 
            vaccine_type: parseInt(vaccineType) 
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('result').innerText = `Predicted Adverse Reaction: ${data.adverse_reaction}`;
    })
    .catch(error => console.error('Error:', error));
});
