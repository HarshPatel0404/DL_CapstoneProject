# Legal Document Classification - Modern UI

## 🚀 Features

### ✨ User Interface Enhancements
- **Splash Screen**: Beautiful animated splash screen with project title and branding
- **Modern Design**: Clean, professional interface with smooth animations
- **Dark/Light Theme**: Toggle between dark and light themes with smooth transitions
- **Model Selection**: Interactive cards to choose between BERT and TF-IDF models
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices

### 🤖 Model Options
1. **Legal-BERT**: Deep learning model fine-tuned on legal documents (Recommended)
2. **TF-IDF + Logistic Regression**: Fast traditional ML approach

### 📊 Visualizations
- **Interactive Gallery**: View all model performance charts
- **Modal View**: Click any visualization to open in full-screen modal
- **Navigation**: Use arrow keys or navigation buttons to browse images
- **Keyboard Shortcuts**: 
  - `ESC` to close modal
  - `←` Previous image
  - `→` Next image

### 🎨 Animations & Effects
- Fade-in animations on page load
- Smooth slide-up effects for cards
- Hover effects on interactive elements
- Loading spinner during predictions
- Animated background particles on splash screen
- Floating icons and bouncing effects

## 🎯 How to Use

### 1. Start the Application
```powershell
python app.py
```

The application will start on `http://localhost:8501`

### 2. Optional: View Splash Screen
Visit `http://localhost:8501/splash` to see the animated splash screen (auto-redirects after 3 seconds)

### 3. Main Application
- **Home Page** (`/`): Main interface for text classification
  - Select your preferred model (BERT or TF-IDF)
  - Paste legal text in the input area
  - Optional: Enable GPU acceleration
  - Optional: Set number of top predictions (BERT only)
  - Click "Analyze Document" to get predictions

### 4. View Visualizations
- Click "Visualizations" in the navigation bar
- Browse all performance metrics and charts
- Click any image to open in full-screen modal
- Use navigation buttons or keyboard arrows to browse

### 5. Theme Toggle
- Click the moon/sun icon in the navigation bar to switch themes
- Theme preference is saved in browser localStorage

## 🎨 Color Schemes

### Light Theme
- Clean white backgrounds
- Purple gradient accents (#667eea → #764ba2)
- Soft shadows and borders

### Dark Theme
- Deep blue-black backgrounds
- Vibrant purple accents
- Enhanced contrast for readability

## 📱 Responsive Features
- Mobile-friendly navigation
- Adaptive grid layouts
- Touch-friendly buttons and controls
- Optimized for all screen sizes

## 🛠️ Technical Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Models**: BERT (Transformers), TF-IDF + Logistic Regression
- **Styling**: Custom CSS with CSS Variables for theming
- **Animations**: CSS keyframe animations

## 📄 File Structure
```
deep learning/
├── app.py                      # Flask application
├── templates/
│   ├── splash.html            # Animated splash screen
│   ├── index.html             # Main home page
│   └── visualizations.html    # Visualizations gallery
├── static/
│   ├── css/
│   │   └── style.css          # Complete styling with themes
│   └── js/
│       └── app.js             # All JavaScript functionality
├── models/                     # Trained models
├── figures/                    # Visualization images
└── README_UI.md               # This file
```

## 🎯 Key Components

### Navigation Bar
- Logo and brand name
- Active page indicator
- Theme toggle button

### Model Selection Cards
- Interactive hover effects
- Visual selection state
- Model badges (Recommended/Fast)

### Input Section
- Large textarea for legal text
- Toggle switches for options
- Responsive number inputs

### Results Display
- Animated result cards
- Confidence bars with gradients
- Color-coded predictions

### Statistics Grid
- Quick overview cards
- Animated icons
- Key metrics display

## 💡 Tips
- Use BERT model for best accuracy
- Dark theme is easier on eyes for long sessions
- Visualizations page shows detailed model performance
- All animations can be viewed by refreshing the page

## 🎨 Customization
To customize colors, edit the CSS variables in `static/css/style.css`:
```css
:root {
    --accent-primary: #667eea;  /* Primary color */
    --accent-secondary: #764ba2; /* Secondary color */
    /* ... more variables ... */
}
```

## 🚀 Future Enhancements
- Export predictions to PDF
- Batch processing of multiple documents
- More visualization types
- User accounts and history
- API documentation page

---

**Enjoy your modern legal document classification system! 🎉**
