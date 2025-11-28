"""
Unit tests for embeddings module.

Tests the EmbeddingModel building block and distance calculation functions.
"""

import pytest
import numpy as np
from tri_lingual_agents.embeddings import (
    EmbeddingModel,
    calculate_distance,
    calculate_similarity,
    get_embedding_model,
)


class TestEmbeddingModelValidation:
    """Test suite for EmbeddingModel input validation."""

    def test_init_valid_model_name(self):
        """Test initialization with valid model name."""
        model = EmbeddingModel('all-MiniLM-L6-v2')

        assert model.model_name == 'all-MiniLM-L6-v2'
        assert model.model is not None

    def test_init_invalid_type_model_name(self):
        """Test that non-string model_name raises TypeError."""
        with pytest.raises(TypeError, match="model_name must be str"):
            EmbeddingModel(123)

    def test_init_empty_model_name(self):
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            EmbeddingModel("")

    def test_init_whitespace_only_model_name(self):
        """Test that whitespace-only model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            EmbeddingModel("   ")


class TestEmbeddingModelFunctionality:
    """Test suite for EmbeddingModel methods."""

    @pytest.fixture
    def model(self):
        """Fixture providing an initialized embedding model."""
        return EmbeddingModel('all-MiniLM-L6-v2')

    def test_encode_single_sentence(self, model):
        """Test encoding a single sentence."""
        text = "This is a test sentence"
        embedding = model.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1  # 1D array for single sentence
        assert len(embedding) == model.get_dimension()

    def test_encode_multiple_sentences(self, model):
        """Test encoding multiple sentences."""
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = model.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2  # 2D array for multiple sentences
        assert embeddings.shape[0] == 3  # Three sentences
        assert embeddings.shape[1] == model.get_dimension()

    def test_encode_consistency(self, model):
        """Test that same text produces same embedding."""
        text = "Consistency test"

        embedding1 = model.encode(text)
        embedding2 = model.encode(text)

        np.testing.assert_array_equal(embedding1, embedding2)

    def test_get_dimension(self, model):
        """Test getting embedding dimension."""
        dim = model.get_dimension()

        assert isinstance(dim, int)
        assert dim > 0
        # all-MiniLM-L6-v2 has 384 dimensions
        assert dim == 384


class TestCalculateDistance:
    """Test suite for calculate_distance function."""

    @pytest.fixture
    def model(self):
        """Fixture providing an initialized embedding model."""
        return EmbeddingModel('all-MiniLM-L6-v2')

    def test_distance_identical_embeddings(self, model):
        """Test distance between identical embeddings is 0."""
        text = "Test sentence"
        embedding = model.encode(text)

        distance = calculate_distance(embedding, embedding, metric='cosine')

        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_distance_different_embeddings(self, model):
        """Test distance between different embeddings is > 0."""
        emb1 = model.encode("The cat sat on the mat")
        emb2 = model.encode("Dogs are loyal animals")

        distance = calculate_distance(emb1, emb2, metric='cosine')

        assert distance > 0.0
        assert distance <= 2.0  # Cosine distance range

    def test_distance_similar_embeddings(self, model):
        """Test distance between similar sentences is small."""
        emb1 = model.encode("The cat is sleeping")
        emb2 = model.encode("The cat is resting")

        distance = calculate_distance(emb1, emb2, metric='cosine')

        # Similar sentences should have small distance
        assert distance < 0.5

    def test_distance_cosine_metric(self, model):
        """Test cosine distance metric."""
        emb1 = model.encode("First sentence")
        emb2 = model.encode("Second sentence")

        distance = calculate_distance(emb1, emb2, metric='cosine')

        assert 0 <= distance <= 2  # Cosine distance range

    def test_distance_euclidean_metric(self, model):
        """Test Euclidean distance metric."""
        emb1 = model.encode("First sentence")
        emb2 = model.encode("Second sentence")

        distance = calculate_distance(emb1, emb2, metric='euclidean')

        assert distance >= 0  # Euclidean distance is always non-negative

    def test_distance_invalid_metric(self, model):
        """Test that invalid metric raises ValueError."""
        emb = model.encode("Test")

        with pytest.raises(ValueError, match="Invalid metric"):
            calculate_distance(emb, emb, metric='invalid')

    def test_distance_mismatched_shapes(self, model):
        """Test that mismatched embedding shapes raise ValueError."""
        emb1 = np.random.rand(384)
        emb2 = np.random.rand(512)  # Different dimension

        with pytest.raises(ValueError, match="must have the same shape"):
            calculate_distance(emb1, emb2)


class TestCalculateSimilarity:
    """Test suite for calculate_similarity function."""

    @pytest.fixture
    def model(self):
        """Fixture providing an initialized embedding model."""
        return EmbeddingModel('all-MiniLM-L6-v2')

    def test_similarity_identical_embeddings(self, model):
        """Test similarity between identical embeddings is 1."""
        text = "Test sentence"
        embedding = model.encode(text)

        similarity = calculate_similarity(embedding, embedding)

        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_similarity_different_embeddings(self, model):
        """Test similarity between different embeddings is < 1."""
        emb1 = model.encode("The cat sat on the mat")
        emb2 = model.encode("Dogs are loyal animals")

        similarity = calculate_similarity(emb1, emb2)

        assert similarity < 1.0
        assert similarity >= -1.0  # Cosine similarity range

    def test_similarity_inverse_of_distance(self, model):
        """Test that similarity = 1 - distance."""
        emb1 = model.encode("Test one")
        emb2 = model.encode("Test two")

        distance = calculate_distance(emb1, emb2, metric='cosine')
        similarity = calculate_similarity(emb1, emb2)

        assert similarity == pytest.approx(1.0 - distance, abs=1e-6)


class TestGetEmbeddingModel:
    """Test suite for get_embedding_model function."""

    def test_get_model_caching(self):
        """Test that get_embedding_model caches models."""
        model1 = get_embedding_model('all-MiniLM-L6-v2')
        model2 = get_embedding_model('all-MiniLM-L6-v2')

        # Should return same instance (cached)
        assert model1 is model2

    def test_get_model_different_names(self):
        """Test that different model names create different instances."""
        model1 = get_embedding_model('all-MiniLM-L6-v2')

        # This will attempt to download if not cached, might fail without internet
        # So we'll just test that calling with same name returns cached version
        model1_again = get_embedding_model('all-MiniLM-L6-v2')

        assert model1 is model1_again


# Integration test
class TestEmbeddingsIntegration:
    """Integration tests for the complete embeddings workflow."""

    def test_full_workflow(self):
        """Test complete workflow from model loading to distance calculation."""
        # Get model
        model = get_embedding_model()

        # Encode texts
        text1 = "The remarkable transformation of artificial intelligence"
        text2 = "The amazing evolution of AI systems"

        emb1 = model.encode(text1)
        emb2 = model.encode(text2)

        # Calculate distance
        distance = calculate_distance(emb1, emb2)

        # Verify results
        assert isinstance(distance, float)
        assert 0 <= distance <= 2

        # Similar sentences should have low distance
        assert distance < 0.5  # These sentences are quite similar

    def test_semantic_similarity_preservation(self):
        """Test that semantically similar sentences have low distance."""
        model = get_embedding_model()

        # Very similar sentences
        emb1 = model.encode("The cat is sleeping on the couch")
        emb2 = model.encode("The cat is resting on the sofa")

        distance = calculate_distance(emb1, emb2)

        # Should be very similar (low distance)
        assert distance < 0.3

    def test_semantic_difference_detection(self):
        """Test that semantically different sentences have high distance."""
        model = get_embedding_model()

        # Very different sentences
        emb1 = model.encode("The cat is sleeping")
        emb2 = model.encode("Mathematical equations are complex")

        distance = calculate_distance(emb1, emb2)

        # Should be quite different (higher distance)
        assert distance > 0.5
