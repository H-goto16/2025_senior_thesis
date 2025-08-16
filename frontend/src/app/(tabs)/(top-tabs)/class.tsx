import { ThemedView } from "@/components/ThemedView";
import { ModelClassManager } from "@/components/model/ModelClassManager";
import { SafeAreaView, ScrollView } from "react-native";

const ModelScreen = () => {
  return (
    <SafeAreaView className="flex-1">
      <ThemedView className="flex-1">
        <ScrollView className="flex-1" showsVerticalScrollIndicator={false}>
          <ModelClassManager />
        </ScrollView>
      </ThemedView>
    </SafeAreaView>
  );
};

export default ModelScreen;