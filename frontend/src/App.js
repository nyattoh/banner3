import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Container, AppBar, Toolbar, Typography, Box } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';

import HomePage from './components/HomePage';
import UploadPage from './components/UploadPage';
import ProcessingPage from './components/ProcessingPage';
import ResultsPage from './components/ResultsPage';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppBar position="static">
          <Toolbar>
            <ImageIcon sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Banner Layer Decomposition
            </Typography>
          </Toolbar>
        </AppBar>
        
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/processing/:imageId" element={<ProcessingPage />} />
            <Route path="/results/:imageId" element={<ResultsPage />} />
          </Routes>
        </Container>
        
        <Box 
          component="footer" 
          sx={{ 
            mt: 'auto', 
            py: 3, 
            px: 2, 
            backgroundColor: 'grey.100',
            textAlign: 'center'
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Banner Layer Decomposition Application - Worker3 Implementation
          </Typography>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;