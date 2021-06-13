import React, {useState} from 'react';
import {View, TouchableOpacity,Text, Alert, Modal, StyleSheet, Pressable,TextInput,useEffect,Dimensions  } from 'react-native';
import {Agenda} from 'react-native-calendars';
import {Card, Avatar} from 'react-native-paper';
import MapView from 'react-native-maps';

const timeToString = (time) => {
 
  const date = new Date(time);
  console.log(date.toISOString().split('T')[0])
  return date.toISOString().split('T')[0];
};

const Schedule: React.FC = () => {
  const [modalVisible, setModalVisible] = useState(false);
  const [resource, setresource] = useState(false);
  const [items, setItems] = useState({});
  const [currentDate, setCurrentDate] = useState('');

  const [currentDay, setCurrentDay] = useState('');
  const [currentMonth, setCurrentMonth] = useState('');
  const [currentYear, setCurrentYear] = useState('');
 
  const postit = (day,month,year) => {
    // setCurrentDay(new Date().getDate())
    // setCurrentMonth(new Date().getMonth())
    // setCurrentYear(new Date().getFullYear())
    // //2017-06-28T00:00:00.000Z

    // if (currentMonth<10){
    //   setCurrentMonth("0"+currentMonth)
    // }
    // if (currentDay<10){
    //   setCurrentDay("0"+currentDay)
    // }
    
    console.log(year+"-"+month+"-"+day+"T00:00:00.000Z")
    items["2021-06-10"].push({
      name: text,
      height: Math.max(50, Math.floor(Math.random() * 150)),
    });

    setModalVisible(false)
    setresource(false)
    console.log(currentDay)
   
  };

  var markers = [
    {
      latitude: 40.65,
      longitude: 74.90,
      title: 'Foo Place',
      subtitle: '1234 Foo Drive'
    }
  ];
  const styles = StyleSheet.create({
    centeredView: {
      flex: 0.1,
      justifyContent: "center",
      alignItems: "center",
      marginTop: 22
    },
    modalView: {
      margin: 20,
      backgroundColor: "white",
      borderRadius: 20,
      padding: 100,
      alignItems: "center",
      shadowColor: "#000",
      shadowOffset: {
        width: 0,
        height: 2
      },
      shadowOpacity: 0.25,
      shadowRadius: 4,
      elevation: 5
    },
    button: {
      borderRadius: 20,
      padding: 10,
      elevation: 2
    },
    buttonOpen: {
      backgroundColor: "#F194FF",
    },
    buttonClose: {
      backgroundColor: "#2196F3",
    },
    textStyle: {
      color: "white",
      fontWeight: "bold",
      textAlign: "center"
    },
    modalText: {
      marginBottom: 15,
      textAlign: "center"
    }
  });
  

  const loadItems = (day) => {
  
    
    setTimeout(() => {
      for (let i = -15; i < 85; i++) {
        const time = day.timestamp + i * 24 * 60 * 60 * 1000;
        const strTime = timeToString(time);
        if (!items[strTime]) {
          items[strTime] = [];
          const numItems = Math.floor(Math.random() * 3 + 1);
          for (let j = 0; j < numItems; j++) {
            items[strTime].push({
              name: 'Item for ' + strTime + ' #' + j,
              height: Math.max(50, Math.floor(Math.random() * 150)),
            });
          }
        }
      }
      const newItems = {};
      Object.keys(items).forEach((key) => {
        newItems[key] = items[key];
      });
      setItems(newItems);
    }, 1000);
  };
  const [text, onChangeText] = React.useState("type");
  const renderItem = (item) => {
    return (
      
      <TouchableOpacity style={{marginRight: 10, marginTop: 17}}>
       
        <Card>
          <Card.Content>
            <View
              style={{
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}>
              <Text>{item.name}</Text>
              <Avatar.Text label="N" />
            </View>
          </Card.Content>
        </Card>
      </TouchableOpacity>
    );
  };


  return (
    
    <View style={{flex: 1}}>
  { resource &&     
<View >
<MapView style={stylesmap.map}
initialRegion={{
  latitude: 37.78825,
  longitude: -122.4324,
  latitudeDelta: 0.0,
  longitudeDelta: 0.0,
}}
>
<MapView.Marker
coordinate={{latitude: 43,
longitude: 79}}
title={"title"}
description={"description"}
/>
 </MapView>
     
    </View>}
      <Agenda
        items={items}
        loadItemsForMonth={loadItems}
        selected={'2021-05-16'}
        renderItem={renderItem}
      />
      <View style={styles.centeredView}>
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => {
          Alert.alert("Modal has been closed.");
          setModalVisible(!modalVisible);
        }}
      >
       
        <View >
          <View style={styles.modalView}>
         
            <Text style={styles.modalText}>Add Input For Today {new Date().getMonth()+1}/{new Date().getDate()}/{new Date().getFullYear()}</Text>
           
            
            <TextInput
        style={styles.input}
        onChangeText={onChangeText}
        value={text}
      />
      <Text></Text>

            <Pressable
              style={[styles.button, styles.buttonClose]}
              onPress={() => postit("06","03","2021")}
            >
              <Text style={styles.textStyle}>Submit!</Text>
            </Pressable>
            <Text></Text>
            <Pressable
              style={[styles.button, styles.buttonClose]}
              onPress={() => setModalVisible(!modalVisible)}
            >
              <Text style={styles.textStyle}>Hide Modal</Text>
            </Pressable>
          </View>
        </View>
      </Modal>


      <Modal
        animationType="slide"
        transparent={true}
        visible={resource}
        onRequestClose={() => {
          Alert.alert("Modal has been closed.");
          setModalVisible(!resource);
        }}
      >
       
        <View >
          
         
           
           
           
           
      <Text></Text>

         
            <Text></Text>
            <Pressable
              style={[styles.button, styles.buttonClose]}
              onPress={() => setresource(!resource)}
            >
              <Text style={styles.textStyle}>Hide Map</Text>
            </Pressable>
            <MapView style={stylesmap.map}
initialRegion={{
  latitude: 42,
  longitude: -79,
  latitudeDelta: 0.0,
  longitudeDelta: 0.0,
}}
>
<MapView.Marker
coordinate={{latitude: 43.7,
  longitude: -79.5}}
title={"Michael Matthew John Cecchini"}
description={"Physician"}
/>

<MapView.Marker
coordinate={{latitude: 43.71,
  longitude: -79.45}}
title={"Jennifer Michelle Salsberg"}
description={"Physician"}
/>

<MapView.Marker
coordinate={{latitude: 43.72,
  longitude: -79.48}}
title={"Sandra Mary Skotnicki Grant"}
description={"Physician"}
/>
 </MapView>
      
          </View>
     
      </Modal>

      <Pressable
        style={[styles.button, styles.buttonOpen]}
        onPress={() => setModalVisible(true)}
      >
        <Text style={styles.textStyle}>Add Input</Text>
      </Pressable>
      <Text></Text>
      <Pressable
        style={[styles.button, styles.buttonOpen]}
        onPress={() => setresource(true)}
      >
        <Text style={styles.textStyle}>Get Help</Text>
      </Pressable>
      <Text></Text>
    </View>
    </View>
  );
   
};

const stylesmap = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  map: {
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height,
  },
});


export default Schedule;