import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  LinearProgress,
  Paper,
  Card,
  CardContent,
  Chip,
  Button,
  Alert,
  Stepper,
  Step,
  StepLabel,
  CircularProgress
} from '@mui/material';
import {
  TextFields,
  Category,
  Landscape,
  CheckCircle,
  Error as ErrorIcon,
  Refresh
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const ProcessingPage = () => {
  const { imageId } = useParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [processingStarted, setProcessingStarted] = useState(false);

  const steps = [
    { label: 'Text Detection', icon: <TextFields />, description: 'Detecting and extracting text elements' },
    { label: 'Object Detection', icon: <Category />, description: 'Identifying objects and products' },
    { label: 'Background Generation', icon: <Landscape />, description: 'Reconstructing clean background' },
    { label: 'Validation', icon: <CheckCircle />, description: 'Validating layer composition' }
  ];

  const getActiveStep = (progress) => {
    if (progress < 20) return 0;
    if (progress < 40) return 1;
    if (progress < 80) return 2;
    if (progress < 100) return 3;
    return 4;
  };

  const fetchStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/status/${imageId}`);
      setStatus(response.data);
      setLoading(false);

      // If processing is complete, redirect to results
      if (response.data.status === 'completed') {
        setTimeout(() => {
          navigate(`/results/${imageId}`);
        }, 2000);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch status');
      setLoading(false);
    }
  };

  const startProcessing = async () => {
    try {
      setError(null);
      await axios.post(`${API_BASE_URL}/api/process/${imageId}`);
      setProcessingStarted(true);
      // Start polling for status updates
      fetchStatus();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start processing');
    }
  };

  useEffect(() => {
    if (!imageId) {
      setError('No image ID provided');
      return;
    }

    // Initial status check
    fetchStatus();

    // Set up polling for status updates
    const interval = setInterval(() => {
      if (status?.status === 'processing' || processingStarted) {
        fetchStatus();
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [imageId, status?.status, processingStarted]);

  const getStatusColor = (statusValue) => {
    switch (statusValue) {
      case 'pending':
        return 'default';
      case 'processing':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (statusValue) => {
    switch (statusValue) {
      case 'processing':
        return <CircularProgress size={20} />;
      case 'completed':
        return <CheckCircle />;
      case 'failed':
        return <ErrorIcon />;
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading status...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Processing Image
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Image ID: {imageId}
      </Typography>

      {/* Error Display */}
      {error && (
        <Alert 
          severity="error" 
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" onClick={() => setError(null)}>
              DISMISS
            </Button>
          }
        >
          {error}
        </Alert>
      )}

      {/* Status Card */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">
              Processing Status
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {getStatusIcon(status?.status)}
              <Chip 
                label={status?.status || 'Unknown'} 
                color={getStatusColor(status?.status)}
                variant="outlined"
              />
            </Box>
          </Box>

          {status?.status === 'pending' && !processingStarted && (
            <Box sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="body1" sx={{ mb: 2 }}>
                Image uploaded successfully. Ready to start processing.
              </Typography>
              <Button
                variant="contained"
                color="primary"
                onClick={startProcessing}
                size="large"
              >
                Start Processing
              </Button>
            </Box>
          )}

          {(status?.status === 'processing' || processingStarted) && (
            <>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {status?.message || 'Processing...'}
              </Typography>
              
              <LinearProgress 
                variant="determinate" 
                value={status?.progress || 0}
                sx={{ mb: 2, height: 8, borderRadius: 4 }}
              />
              
              <Typography variant="body2" textAlign="center">
                {status?.progress || 0}% Complete
              </Typography>
            </>
          )}

          {status?.status === 'completed' && (
            <Box sx={{ textAlign: 'center', py: 2 }}>
              <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
              <Typography variant="h6" color="success.main" gutterBottom>
                Processing Complete!
              </Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Redirecting to results page...
              </Typography>
              <Button
                variant="contained"
                color="primary"
                onClick={() => navigate(`/results/${imageId}`)}
              >
                View Results
              </Button>
            </Box>
          )}

          {status?.status === 'failed' && (
            <Box sx={{ textAlign: 'center', py: 2 }}>
              <ErrorIcon sx={{ fontSize: 48, color: 'error.main', mb: 2 }} />
              <Typography variant="h6" color="error.main" gutterBottom>
                Processing Failed
              </Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                {status?.error || 'An unknown error occurred'}
              </Typography>
              <Button
                variant="outlined"
                color="primary"
                startIcon={<Refresh />}
                onClick={startProcessing}
              >
                Retry Processing
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Processing Steps */}
      {(status?.status === 'processing' || status?.status === 'completed') && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Processing Steps
          </Typography>
          
          <Stepper 
            activeStep={getActiveStep(status?.progress || 0)} 
            orientation="vertical"
          >
            {steps.map((step, index) => (
              <Step key={step.label}>
                <StepLabel
                  icon={step.icon}
                  optional={
                    <Typography variant="caption" color="text.secondary">
                      {step.description}
                    </Typography>
                  }
                >
                  {step.label}
                </StepLabel>
              </Step>
            ))}
          </Stepper>
        </Paper>
      )}

      {/* Navigation */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          variant="outlined"
          onClick={() => navigate('/upload')}
        >
          Upload Another Image
        </Button>
        
        {status?.status === 'completed' && (
          <Button
            variant="contained"
            onClick={() => navigate(`/results/${imageId}`)}
          >
            View Results
          </Button>
        )}
      </Box>
    </Box>
  );
};

export default ProcessingPage;