import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  CloudUpload,
  TextFields,
  Category,
  Landscape,
  Download,
  CheckCircle
} from '@mui/icons-material';

const HomePage = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <TextFields color="primary" />,
      title: 'Text Layer Extraction',
      description: 'Automatically detect and extract text elements with hybrid OCR technology'
    },
    {
      icon: <Category color="primary" />,
      title: 'Object Detection',
      description: 'Identify and isolate products, people, and other objects using YOLOv8'
    },
    {
      icon: <Landscape color="primary" />,
      title: 'Background Reconstruction',
      description: 'Generate clean backgrounds using advanced inpainting algorithms'
    },
    {
      icon: <Download color="primary" />,
      title: 'Layer Export',
      description: 'Download individual layers or complete sets as PNG files'
    }
  ];

  const steps = [
    'Upload your banner image (PNG, JPG)',
    'Wait for automatic processing',
    'Review the extracted layers',
    'Download individual layers or ZIP package'
  ];

  return (
    <Box>
      {/* Hero Section */}
      <Paper
        sx={{
          position: 'relative',
          backgroundColor: 'grey.800',
          color: '#fff',
          mb: 4,
          backgroundSize: 'cover',
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'center',
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            right: 0,
            left: 0,
            backgroundColor: 'rgba(0,0,0,.3)',
          }}
        />
        <Grid container>
          <Grid item md={8}>
            <Box
              sx={{
                position: 'relative',
                p: { xs: 3, md: 6 },
                pr: { md: 0 },
              }}
            >
              <Typography component="h1" variant="h3" color="inherit" gutterBottom>
                Banner Image Layer Decomposition
              </Typography>
              <Typography variant="h5" color="inherit" paragraph>
                Automatically separate your banner images into text, object, and background layers
                using advanced AI technology.
              </Typography>
              <Button
                variant="contained"
                color="secondary"
                size="large"
                startIcon={<CloudUpload />}
                onClick={() => navigate('/upload')}
                sx={{ mt: 2 }}
              >
                Start Processing
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Features Section */}
      <Typography variant="h4" component="h2" gutterBottom sx={{ mb: 4 }}>
        Key Features
      </Typography>
      
      <Grid container spacing={4} sx={{ mb: 6 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1, textAlign: 'center' }}>
                <Box sx={{ mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography gutterBottom variant="h6" component="h3">
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* How It Works Section */}
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Typography variant="h4" component="h2" gutterBottom>
            How It Works
          </Typography>
          <List>
            {steps.map((step, index) => (
              <ListItem key={index} sx={{ pl: 0 }}>
                <ListItemIcon>
                  <CheckCircle color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary={`${index + 1}. ${step}`}
                  sx={{ ml: 1 }}
                />
              </ListItem>
            ))}
          </List>
          <Button
            variant="outlined"
            color="primary"
            size="large"
            onClick={() => navigate('/upload')}
            sx={{ mt: 2 }}
          >
            Get Started
          </Button>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Technology Stack
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Text Detection:</strong> Hybrid OCR using Tesseract and EasyOCR
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Object Detection:</strong> YOLOv8 with background removal
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Background Inpainting:</strong> Advanced multi-scale algorithms
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Quality Validation:</strong> SSIM and PSNR metrics
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default HomePage;