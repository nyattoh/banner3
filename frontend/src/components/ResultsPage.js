import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardMedia,
  Grid,
  Button,
  Chip,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress
} from '@mui/material';
import {
  Download,
  Archive,
  Visibility,
  ExpandMore,
  CheckCircle,
  Error as ErrorIcon,
  Info
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const ResultsPage = () => {
  const { imageId } = useParams();
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTab, setSelectedTab] = useState(0);
  const [previewImages, setPreviewImages] = useState({});

  useEffect(() => {
    if (!imageId) {
      setError('No image ID provided');
      return;
    }

    fetchResults();
  }, [imageId]);

  const fetchResults = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/results/${imageId}`);
      setResults(response.data);
      setLoading(false);

      // Load preview images
      loadPreviewImages();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch results');
      setLoading(false);
    }
  };

  const loadPreviewImages = () => {
    const layers = ['original', 'text', 'object', 'background', 'composed'];
    
    layers.forEach(layer => {
      const img = new Image();
      img.onload = () => {
        setPreviewImages(prev => ({
          ...prev,
          [layer]: `${API_BASE_URL}/api/preview/${imageId}/${layer}`
        }));
      };
      img.onerror = () => {
        console.warn(`Failed to load preview for ${layer} layer`);
      };
      img.src = `${API_BASE_URL}/api/preview/${imageId}/${layer}`;
    });
  };

  const handleDownload = async (layerType) => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/download/${imageId}/${layerType}`,
        { responseType: 'blob' }
      );
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      const filename = response.headers['content-disposition']?.split('filename=')[1]?.replace(/"/g, '') ||
                     `${imageId}_${layerType}.png`;
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
      setError('Failed to download file');
    }
  };

  const handleDownloadAll = () => {
    handleDownload('all');
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const LayerCard = ({ layer, title, description, color }) => (
    <Card sx={{ height: '100%' }}>
      <CardMedia
        component="img"
        height="200"
        image={previewImages[layer] || '/placeholder-image.png'}
        alt={`${title} layer`}
        sx={{ objectFit: 'contain', backgroundColor: 'grey.100' }}
      />
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="h6" component="h3">
            {title}
          </Typography>
          <Chip label={layer} color={color} size="small" />
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {description}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            size="small"
            variant="outlined"
            startIcon={<Visibility />}
            onClick={() => window.open(previewImages[layer], '_blank')}
            disabled={!previewImages[layer]}
          >
            Preview
          </Button>
          <Button
            size="small"
            variant="contained"
            startIcon={<Download />}
            onClick={() => handleDownload(layer)}
          >
            Download
          </Button>
        </Box>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading results...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ maxWidth: 800, mx: 'auto' }}>
        <Alert 
          severity="error" 
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" onClick={() => navigate('/upload')}>
              Upload New Image
            </Button>
          }
        >
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Processing Results
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Image ID: {imageId}
          </Typography>
        </Box>
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<Archive />}
          onClick={handleDownloadAll}
        >
          Download All Layers
        </Button>
      </Box>

      {/* Validation Status */}
      {results?.validation && (
        <Alert
          severity={results.validation.is_valid ? 'success' : 'warning'}
          icon={results.validation.is_valid ? <CheckCircle /> : <ErrorIcon />}
          sx={{ mb: 3 }}
        >
          <Box>
            <Typography variant="subtitle2">
              Layer Composition Validation: {results.validation.is_valid ? 'Passed' : 'Issues Detected'}
            </Typography>
            <Typography variant="body2">
              Similarity Score: {(results.validation.similarity_score * 100).toFixed(1)}%
            </Typography>
            {results.validation.errors?.length > 0 && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                Issues: {results.validation.errors.join(', ')}
              </Typography>
            )}
          </Box>
        </Alert>
      )}

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={selectedTab} onChange={(e, newValue) => setSelectedTab(newValue)}>
          <Tab label="Layer Preview" />
          <Tab label="Quality Metrics" />
          <Tab label="Detection Details" />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {selectedTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <LayerCard
              layer="original"
              title="Original Image"
              description="The original uploaded banner image"
              color="default"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <LayerCard
              layer="composed"
              title="Composed Result"
              description="All layers combined back together for validation"
              color="primary"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <LayerCard
              layer="text"
              title="Text Layer"
              description="Extracted text elements with transparency"
              color="secondary"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <LayerCard
              layer="object"
              title="Object Layer"
              description="Detected objects and products with transparency"
              color="info"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <LayerCard
              layer="background"
              title="Background Layer"
              description="Clean background with text and objects removed"
              color="success"
            />
          </Grid>
        </Grid>
      )}

      {selectedTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Quality Metrics
                </Typography>
                {results?.quality_metrics && (
                  <TableContainer>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell><strong>SSIM Score</strong></TableCell>
                          <TableCell>{results.quality_metrics.ssim?.toFixed(4) || 'N/A'}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>PSNR (dB)</strong></TableCell>
                          <TableCell>{results.quality_metrics.psnr?.toFixed(2) || 'N/A'}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>MSE</strong></TableCell>
                          <TableCell>{results.quality_metrics.mse?.toFixed(2) || 'N/A'}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Processing Information
                </Typography>
                {results?.processing_result && (
                  <TableContainer>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell><strong>Processing Time</strong></TableCell>
                          <TableCell>{results.processing_result.processing_time?.toFixed(2)} seconds</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>Completed At</strong></TableCell>
                          <TableCell>
                            {new Date(results.processing_result.completed_at).toLocaleString()}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell><strong>Total Layers</strong></TableCell>
                          <TableCell>{results.processing_result.layers?.length || 0}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {selectedTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Text Detection Results</Typography>
              </AccordionSummary>
              <AccordionDetails>
                {results?.detection_summary?.text_detection && (
                  <>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      <strong>Regions Detected:</strong> {results.detection_summary.text_detection.regions}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Detected Text:</strong>
                    </Typography>
                    {results.detection_summary.text_detection.detected_text?.map((text, index) => (
                      <Chip
                        key={index}
                        label={text}
                        variant="outlined"
                        size="small"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </>
                )}
              </AccordionDetails>
            </Accordion>
          </Grid>
          <Grid item xs={12} md={6}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">Object Detection Results</Typography>
              </AccordionSummary>
              <AccordionDetails>
                {results?.detection_summary?.object_detection && (
                  <>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      <strong>Objects Detected:</strong> {results.detection_summary.object_detection.regions}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Object Classes:</strong>
                    </Typography>
                    {results.detection_summary.object_detection.detected_classes?.map((className, index) => (
                      <Chip
                        key={index}
                        label={className}
                        variant="outlined"
                        size="small"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </>
                )}
              </AccordionDetails>
            </Accordion>
          </Grid>
        </Grid>
      )}

      {/* Navigation */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          variant="outlined"
          onClick={() => navigate('/upload')}
        >
          Process Another Image
        </Button>
        <Button
          variant="outlined"
          onClick={() => navigate('/')}
        >
          Back to Home
        </Button>
      </Box>
    </Box>
  );
};

export default ResultsPage;